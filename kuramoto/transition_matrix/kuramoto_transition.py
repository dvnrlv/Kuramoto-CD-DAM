"""
Transition-Matrix Extension of the High-Order Kuramoto Associative Memory
==========================================================================

kuramoto_transition.py
│
├─ [1-47]    docstring + imports        ── 6 names from kuramoto_library2
│
├─ [56-102]  BUILDERS  (pure numpy, no state)
│              transition_from_function(P, fn)      → (P,P)
│              static_transition(P, gamma)          → (P,P)
│              shift_transition(P, nhop, boundary)  → (P,P)
│              mixed_transition(P, gamma, kappa)    → (P,P)
│
├─ [109-190] KERNELS  (@njit, no self, no Python objects)
│              _transition_drive(cos_theta, xi, T, d, V)      → fills V
│              _simulate_transition(theta_init, omega, xi, T,
│                                   d, dt, num_steps,
│                                   phase_noise, seed)        → history
│
└─ [197-262] NETWORK
               class TransitionNetwork(KuramotoNetwork)
                 __init__     stores self.transition
                 _integrate   THE ONLY OVERRIDE

                 
                 

`kuramoto_library2` hard-wires "mu drives mu+1" into the successor tensor `xi_next`.
This module replaces that fixed chain with a transition tensor T the caller supplies:

    dtheta_i/dt = omega_i - sin(theta_i) * sum_a sum_alpha xi_i^{a,alpha}
                                          sum_beta T^a_{alpha beta} (m_i^{a,beta})^d

    m_i^{a,beta} = (1/(N-1)) sum_{j!=i} xi_j^{a,beta} cos(theta_j)

T^a_{alpha beta} is how strongly pattern beta drives pattern alpha within sequence a --
row is the destination, column the source. Everything else is `kuramoto_library2`:
storage, generation, diagnostics, decoders and plotting are inherited unchanged, and
the kernel is `_multi_drive` with the single `xi_next` term replaced by a sum over
alpha. T is (K, P, P) alongside the (K, P, N) `MultiSequence.xi_tensor()`, or a lone
(P, P) matrix shared by every sequence.

`shift_transition` recovers the `xi_next` dynamics exactly, so it is the baseline.
There is no `boundary=` argument: what the last pattern points at is column P-1 of T.

    import kuramoto_library2 as kl
    from kuramoto_transition import TransitionNetwork, mixed_transition

    CONFIG = dict(dt=0.02, T=55.0, frequency_mean=0.0, frequency_std=0.03, phase_noise=0.01,
                  d=3, corruption_rate=0.2, tolerance=0.4, mode="ode")
    multi = kl.generate_multi_sequence(N=40, seq_len=5, corruption_rate=0.2, count=2, seed=10)
    net = TransitionNetwork(transition=mixed_transition(len(multi[0]), gamma=0.3),
                            N=40, seed=10, **CONFIG)
    net.plot(net.simulate(multi), multi)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from kuramoto_library2 import (
    KuramotoNetwork,
    MultiSequence,
    STATIONARY_EPS,
    STATIONARY_PATIENCE,
    _default,
    njit,
)


# ---------------------------------------------------------------------------
# Transition matrices -- free functions, like the generation functions in
# kuramoto_library2. Each returns a plain (P, P) array; combine them with
# ordinary arithmetic (`A + 0.5 * B`) and stack per-sequence rules with np.stack.
# ---------------------------------------------------------------------------

def transition_from_function(P: int, fn) -> np.ndarray:
    """T[alpha, beta] = fn(alpha, beta), for an arbitrary coupling."""
    return np.array([[float(fn(a, b)) for b in range(P)] for a in range(P)])


def static_transition(P: int, gamma: float = 1.0) -> np.ndarray:
    """gamma * I: every pattern drives only itself -- P fixed points, no sequence.

    The auto-associative Krotov DAM case, and the only family that stays a gradient
    flow at d > 1, which makes it the natural stabilizing term to add to a chain.
    """
    return gamma * np.eye(P)


def shift_transition(P: int, nhop: int = 1, boundary: Optional[str] = None,
                     strength: float = 1.0) -> np.ndarray:
    """strength * Sigma: pattern beta drives the pattern `nhop` steps after it.

    The matrix form of `Sequence.xi_hop`, resolving the end of the chain the same two
    ways -- "self" (default) clamps, so T[P-1, P-1] holds and the trajectory settles;
    "cycle" wraps mod P, so it loops forever. With nhop=1 those differ in exactly one
    entry, and either can also be reached by editing the returned array.
    """
    rows = np.arange(P) + nhop
    boundary = _default(boundary, "self")
    if boundary == "self":
        rows = np.clip(rows, 0, P - 1)
    elif boundary == "cycle":
        rows %= P
    else:
        raise ValueError("boundary must be 'self' or 'cycle'")

    T = np.zeros((P, P))
    T[rows, np.arange(P)] = strength
    return T


def mixed_transition(P: int, gamma: float = 1.0, kappa: float = 1.0,
                     nhop: int = 1, boundary: Optional[str] = None) -> np.ndarray:
    """gamma * I + kappa * Sigma: dwell plus advance.

    The diagonal part is symmetric and makes each pattern something to sit in; Sigma
    is what drives traversal. gamma/kappa is therefore the dwell-time knob, and
    gamma = 0 is the bare sequence dynamics.
    """
    return (static_transition(P, gamma)
            + shift_transition(P, nhop=nhop, boundary=boundary, strength=kappa))


# ---------------------------------------------------------------------------
# Low-level numba kernels (free functions -- numba methods can't take `self`)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _transition_drive(cos_theta, xi, T, d, V):
    """V[i] = sum_a sum_alpha xi_i^{a,alpha} sum_beta T^a_{alpha beta} (m_i^{a,beta})^d.

    `_multi_drive` with its one `xi_next` term replaced by a loop over alpha. Sharpening
    each (a, beta) overlap into `h` before that loop keeps the power count identical to
    the base kernel, however dense T is; zero entries are skipped, so a sparse T (Sigma
    has P nonzeros) costs no more than the fixed successor chain.
    """
    K, P, N = xi.shape
    h = np.empty(N, dtype=np.float64)

    for i in range(N):
        V[i] = 0.0

    for a in range(K):
        for beta in range(P):
            overlap = 0.0
            for j in range(N):
                overlap += xi[a, beta, j] * cos_theta[j]
            for i in range(N):
                inner_sum = overlap - xi[a, beta, i] * cos_theta[i]   # drop the j == i term
                h[i] = (inner_sum / (N - 1)) ** d

            for alpha in range(P):
                weight = T[a, alpha, beta]
                if weight == 0.0:
                    continue
                for i in range(N):
                    V[i] += xi[a, alpha, i] * weight * h[i]


@njit(cache=True, fastmath=True)
def _simulate_transition(theta_init, omega, xi, T, d, dt, num_steps, phase_noise, seed):
    """Euler/Euler-Maruyama loop and Stationary Break Check of `_simulate_euler_multi`,
    driven by `_transition_drive`. phase_noise = 0 is the ODE, > 0 the SDE.

    Stationarity is judged on the deterministic drift alone in both cases: once that has
    settled, further integration is just wandering near a fixed point.
    """
    np.random.seed(seed)
    N = theta_init.shape[0]

    history = np.empty((num_steps + 1, N), dtype=np.float64)
    theta = theta_init.copy()
    history[0] = theta

    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    V = np.empty(N, dtype=np.float64)

    noise_scale = phase_noise * np.sqrt(dt)
    stationary_count = 0
    actual_steps = num_steps

    for step in range(num_steps):
        for i in range(N):
            cos_theta[i] = np.cos(theta[i])
            sin_theta[i] = np.sin(theta[i])

        _transition_drive(cos_theta, xi, T, d, V)

        max_abs_dtheta = 0.0
        for i in range(N):
            dtheta_i = omega[i] - sin_theta[i] * V[i]
            theta[i] = theta[i] + dt * dtheta_i
            if noise_scale > 0.0:
                theta[i] = theta[i] + noise_scale * np.random.randn()
            if abs(dtheta_i) > max_abs_dtheta:
                max_abs_dtheta = abs(dtheta_i)

        history[step + 1] = theta.copy()

        if max_abs_dtheta < STATIONARY_EPS:
            stationary_count += 1
            if stationary_count >= STATIONARY_PATIENCE:
                actual_steps = step + 1
                break
        else:
            stationary_count = 0

    return history[:actual_steps + 1]


# ---------------------------------------------------------------------------
# Network: KuramotoNetwork with the transition-matrix coupling swapped in
# ---------------------------------------------------------------------------

class TransitionNetwork(KuramotoNetwork):
    """`KuramotoNetwork` driven by a transition matrix instead of the successor chain.

    Only the coupling changes, so `_integrate` is the only override: `simulate` and every
    diagnostic (`overlaps`, both decoders, `compare_to_stored_sequence`, `plot`) are
    inherited verbatim. T is fixed per network, like every other run parameter, so
    sweeping it means constructing a new `TransitionNetwork` per matrix.
    """

    def __init__(self, transition, **kwargs):
        """`transition` is a (P, P) matrix shared by every sequence, or a (K, P, P)
        tensor giving each its own. Remaining kwargs are the base config."""
        super().__init__(**kwargs)
        self.transition = transition

    def _integrate(
        self,
        sequences,
        theta_init: np.ndarray,
        mode: Optional[str],
        seed: Optional[int],
        T: Optional[float],
        boundary: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Same contract as `KuramotoNetwork._integrate`. A lone `Sequence` is promoted
        to the K = 1 case rather than given its own kernel, so the sum over `a` is a
        one-iteration loop and there is one implementation of the rule.

        `T` here is the inherited *duration* argument, not the transition matrix.
        """
        if boundary is not None:
            raise ValueError(
                "boundary belongs to the xi_next dynamics; with a transition matrix, what "
                "the last pattern points at is column P-1 of T (see shift_transition)."
            )

        mode = _default(mode, self.mode)
        num_steps = self.num_steps_for(_default(T, self.T))

        xi = sequences.xi_tensor() if isinstance(sequences, MultiSequence) else sequences.xi()[np.newaxis]
        xi = np.ascontiguousarray(xi)
        K, P = xi.shape[:2]

        transition = np.asarray(self.transition, dtype=np.float64)
        if transition.shape == (P, P):
            transition = np.broadcast_to(transition, (K, P, P))
        if transition.shape != (K, P, P):
            raise ValueError(
                f"Transition must be ({P}, {P}) or ({K}, {P}, {P}) for {K} sequences of "
                f"{P} patterns, got {transition.shape}."
            )
        # Materialized rather than left as a stride-0 broadcast view: numba specializes
        # on layout.
        transition = np.ascontiguousarray(transition)

        if mode not in ("ode", "sde"):
            raise ValueError("mode must be either 'ode' or 'sde'")

        theta_history = _simulate_transition(
            theta_init, self.omega, xi, transition, self.d, self.dt, num_steps,
            self.phase_noise if mode == "sde" else 0.0, _default(seed, self.seed),
        )

        actual_steps = theta_history.shape[0] - 1
        time = np.linspace(0.0, actual_steps * self.dt, actual_steps + 1)
        return theta_history, time


# ---------------------------------------------------------------------------
# Example usage (only runs when executed directly, not on import)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import kuramoto_library2 as kl

    CONFIG = dict(dt=0.02, T=55.0, frequency_mean=0.0, frequency_std=0.03, phase_noise=0.01,
                  d=3, corruption_rate=0.2, tolerance=0.4, mode="ode")

    multi = kl.generate_multi_sequence(N=40, seq_len=5, corruption_rate=0.2, count=2, seed=10)
    P = len(multi[0])

    # gamma * I + Sigma, cycling: each sequence dwells on a pattern, advances, and loops.
    net = TransitionNetwork(transition=mixed_transition(P, gamma=0.3, boundary="cycle"),
                            N=40, seed=10, **CONFIG)
    result = net.simulate(multi)
    print(net.decode_sequence_peaks(result, multi, min_distance=50))
    net.plot(result, multi)
