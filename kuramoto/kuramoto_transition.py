"""
Transition-Matrix Extension of the High-Order Kuramoto Associative Memory
==========================================================================

Extends `kuramoto_library2` without modifying it: everything here is new, and the
storage classes, diagnostics, decoders and plotting are inherited unchanged.

Model
-----
`kuramoto_library2` hard-wires one pattern-to-pattern coupling -- "mu drives mu+1,
weight 1" -- into the successor tensor `xi_next`. This module replaces that fixed chain
with an explicit transition matrix T^a per stored sequence, so that

    dtheta_i/dt = omega_i - sin(theta_i) * sum_{a=1}^{K} sum_{alpha=1}^{P} xi_i^{a,alpha}
                                           sum_{beta=1}^{P} T^a_{alpha beta} (m_i^{a,beta})^d

    m_i^{a,beta} = (1/(N-1)) sum_{j!=i} xi_j^{a,beta} cos(theta_j)

Reading the indices: `T^a_{alpha beta}` is how strongly overlap with pattern beta of
sequence a drives pattern alpha of that same sequence. The sum over `a` is the outer sum
over stored sequences, exactly as in the multi-sequence rule; the sums over alpha and
beta replace the single sum over mu, which is recovered as the special case
T^a_{alpha beta} = 1 iff alpha == beta + 1 (`shift_transition`).

T is therefore an order-3 array of shape (K, P, P), one (P, P) matrix per sequence,
sitting alongside the (K, P, N) pattern tensor `MultiSequence.xi_tensor()` that
`kuramoto_library2` already builds -- the sequence index `a` is axis 0 of both, so the
triple sum above is a plain nested loop over axes in `_transition_drive`. Because T^a is
indexed within one sequence, patterns never drive patterns of a *different* sequence;
that is what "K independent stored sequences" means.

Choosing T
----------
Any (P, P) array works -- nothing here inspects how it was built:

    static_transition(P, gamma)          gamma * delta_{alpha beta}: no sequence, P fixed
                                         points (the auto-associative / Krotov DAM case)
    shift_transition(P, ...)             Sigma: the Sompolinsky-Kanter successor chain,
                                         reproducing the `xi_next` dynamics exactly
    mixed_transition(P, gamma, kappa)    gamma*I + kappa*Sigma: dwell plus advance
    transition_from_function(P, fn)      T[alpha, beta] = fn(alpha, beta), anything at all

`stack_transitions` lifts these to the (K, P, P) tensor, either one rule for every
sequence or a per-sequence list. A lone (P, P) matrix passed to `TransitionNetwork` is
broadcast to all K sequences, and a builder is called with P, so the common cases need
no stacking by hand.

Where the last pattern points
-----------------------------
There is no `boundary=` argument here: what pattern P-1 transitions to is a property of
T, not of the integrator, and lives in column P-1 of the matrix.
`shift_transition(P, boundary=...)` sets it for you --

    boundary="self"  (default)   T[P-1, P-1] = 1   last pattern drives itself, the
                                                   trajectory settles there
    boundary="cycle"             T[0,   P-1] = 1   last pattern drives the first,
                                                   the sequence loops forever

-- and since it is a single entry, it can also just be edited on the returned array.
See `shift_transition` for the per-sequence version of the same toggle.

Typical usage
-------------
    import kuramoto_library2 as kl
    from kuramoto_transition import TransitionNetwork, shift_transition, mixed_transition

    CONFIG = dict(dt=0.02, T=55.0, frequency_std=0.03, phase_noise=0.01,
                  d=3, corruption_rate=0.2, tolerance=0.4, mode="ode")

    multi = kl.generate_multi_sequence(N=40, seq_len=5, corruption_rate=0.2, count=2, seed=10)
    T = mixed_transition(len(multi[0]), gamma=0.3, kappa=1.0, boundary="cycle")

    net = TransitionNetwork(transition=T, N=40, seed=10, **CONFIG)
    result = net.simulate(multi)
    net.plot(result, multi)

Sweeping T across trials means constructing a new `TransitionNetwork` per matrix while
reusing the same CONFIG, the same way `kuramoto_library2` sweeps N or seed.
"""

from __future__ import annotations

from typing import Optional, Tuple
from typing import Sequence as TypingSequence

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
# kuramoto_library2 (and for the same reason: only they know how to build a T)
# ---------------------------------------------------------------------------

def transition_from_function(P: int, fn) -> np.ndarray:
    """T[alpha, beta] = fn(alpha, beta), for an arbitrary coupling.

    `fn` takes two 0-based pattern indices and returns a scalar, so the whole family is
    one lambda wide:

        transition_from_function(P, lambda a, b: 1.0 if a == b + 1 else 0.0)   # Sigma
        transition_from_function(P, lambda a, b: 0.5 ** abs(a - b))            # decaying
        transition_from_function(P, lambda a, b: 1.0 if a == (b + 2) % P else 0.0)

    Built index by index rather than vectorized: P is single digits to low tens, so `fn`
    stays an ordinary scalar function of (alpha, beta) and reads like the sum it defines.
    """
    return np.array([[float(fn(alpha, beta)) for beta in range(P)] for alpha in range(P)])


def static_transition(P: int, gamma: float = 1.0) -> np.ndarray:
    """T[alpha, beta] = gamma * delta_{alpha beta}: every pattern drives only itself.

    The static case -- no sequence at all, just P fixed-point attractors, recovering the
    polynomial Krotov-Hopfield DAM dynamics. Diagonal T is also the one family that stays
    a genuine gradient flow at d > 1, which makes it the natural stabilizing term to add
    to a sequence matrix (see `mixed_transition`).
    """
    return gamma * np.eye(P)


def shift_transition(P: int, nhop: int = 1, boundary: Optional[str] = None,
                     strength: float = 1.0) -> np.ndarray:
    """T = strength * Sigma: pattern beta drives the pattern `nhop` steps after it, so
    the nonzero entries are T[beta + nhop, beta].

    The matrix form of `Sequence.xi_hop`, resolving the end of the chain the same two
    ways -- this is the cycle/self toggle, and it lives here rather than on the
    integrator because "what does the last pattern point at" is a statement about T:

    - None (default) or "self": clamped, rows that would run past the end pile up on the
      last pattern, so T[P-1, P-1] picks up their weight and the trajectory settles there.
    - "cycle": taken mod P, beta -> (beta + nhop) % P, a circulant that loops forever.

    With nhop=1 the two differ in exactly one entry -- "self" puts the 1 of column P-1 on
    row P-1, "cycle" puts it on row 0 -- so either can also be reached by editing the
    returned array:

        T = shift_transition(P)      # settles at the last pattern
        T[P-1, P-1], T[0, P-1] = 0.0, 1.0   # ... now it cycles instead

    Weights are accumulated with `+=` rather than assigned, so the clamped case sums
    correctly when several source patterns near the end all point at the terminal one.
    """
    boundary = _default(boundary, "self")
    rows = np.arange(P)
    if boundary == "self":
        rows = np.clip(rows + nhop, 0, P - 1)
    elif boundary == "cycle":
        rows = (rows + nhop) % P
    else:
        raise ValueError("boundary must be 'self' or 'cycle'")

    T = np.zeros((P, P))
    for beta in range(P):
        T[rows[beta], beta] += strength
    return T


def mixed_transition(P: int, gamma: float = 1.0, kappa: float = 1.0,
                     nhop: int = 1, boundary: Optional[str] = None) -> np.ndarray:
    """T = gamma * I + kappa * Sigma: static self-coupling plus sequence advance.

    The two terms pull in opposite directions -- the diagonal part is symmetric and
    stabilizes, making each pattern something to dwell in, while Sigma is what drives
    one-directional traversal -- so gamma/kappa is the dwell-time knob and gamma = 0 is
    the bare sequence dynamics with no restoring term.
    """
    return static_transition(P, gamma) + shift_transition(P, nhop=nhop, boundary=boundary,
                                                          strength=kappa)


def stack_transitions(build, K: int, P: int, **kwargs) -> np.ndarray:
    """The (K, P, P) tensor the kernels take: T[a] is sequence a's own matrix, in the
    same order as `MultiSequence.xi_tensor()`.

    `build` is one builder applied to every sequence (called as `build(P, **kwargs)`), or
    a list of K entries -- builders or ready-made (P, P) arrays -- to give each sequence
    its own rule, e.g. one cycling while the other settles:

        stack_transitions(shift_transition, K, P, boundary="cycle")
        stack_transitions([shift_transition, static_transition], K, P)
    """
    builds = list(build) if isinstance(build, (list, tuple)) else [build] * K
    if len(builds) != K:
        raise ValueError(f"Got {len(builds)} transition rules for {K} sequences.")
    return np.stack([
        rule(P, **kwargs) if callable(rule) else np.asarray(rule, dtype=np.float64)
        for rule in builds
    ])


def _resolve_transition(transition, K: int, P: int) -> np.ndarray:
    """Normalize whatever the caller supplied into a contiguous float64 (K, P, P) array.

    Accepts the tensor itself, a single (P, P) matrix (broadcast to every sequence -- the
    common case of "all sequences share one rule"), a builder called with P, or a
    per-sequence list of either. Broadcasting is materialized with a copy rather than
    left as a stride-0 view, since numba specializes on layout."""
    if callable(transition) or isinstance(transition, (list, tuple)):
        transition = stack_transitions(transition, K, P)

    T = np.asarray(transition, dtype=np.float64)
    if T.shape == (P, P):
        T = np.broadcast_to(T, (K, P, P))
    if T.shape != (K, P, P):
        raise ValueError(
            f"Transition must be ({P}, {P}) or ({K}, {P}, {P}) for {K} sequences of "
            f"{P} patterns, got {T.shape}."
        )
    return np.ascontiguousarray(T)


# ---------------------------------------------------------------------------
# Low-level numba kernels (free functions -- numba methods can't take `self`)
# ---------------------------------------------------------------------------
#
# `xi` is the (K, P, N) tensor from `MultiSequence.xi_tensor()` and `T` the matching
# (K, P, P) transition tensor, so the sums over a, alpha, beta are a nested loop over
# their shared first two axes -- the same layout `_multi_drive` uses for sum_a sum_mu.

@njit(cache=True, fastmath=True)
def _transition_drive(cos_theta, xi, T, d, m, V):
    """V[i] = sum_a sum_alpha xi_i^{a,alpha} sum_beta T^a_{alpha beta} (m_i^{a,beta})^d.

    `m` is a caller-owned (K, P, N) scratch buffer, allocated once per run rather than
    per step: unlike `_multi_drive`, which consumes each pattern's overlap immediately,
    every overlap here can be read back by any number of alphas through T, so all of them
    have to exist at once."""
    K, P, N = xi.shape

    for i in range(N):
        V[i] = 0.0

    # Sharpened leave-one-out overlaps (m_i^{a,beta})^d, one row per (a, beta): a single
    # full overlap plus the cheap j == i subtraction, as in the kernels of the base
    # module. Raised to the power here, once, since every alpha that reads pattern
    # (a, beta) through T reads the same sharpened value.
    for a in range(K):
        for beta in range(P):
            overlap = 0.0
            for j in range(N):
                overlap += xi[a, beta, j] * cos_theta[j]
            for i in range(N):
                inner_sum = overlap - xi[a, beta, i] * cos_theta[i]   # drop the j == i term
                m[a, beta, i] = (inner_sum / (N - 1)) ** d

    # sum_a sum_alpha sum_beta. Zero entries of T are skipped, which is what keeps the
    # nominally P^2 mixing as cheap as the old sum over mu whenever T is sparse -- as it
    # is for Sigma, which has exactly P nonzeros.
    for a in range(K):
        for alpha in range(P):
            for beta in range(P):
                if T[a, alpha, beta] == 0.0:
                    continue
                weight = T[a, alpha, beta]
                for i in range(N):
                    V[i] += xi[a, alpha, i] * weight * m[a, beta, i]


@njit(cache=True, fastmath=True)
def _simulate_euler_transition(theta_init, omega, xi, T, d, dt, num_steps,
                                stationary_eps=STATIONARY_EPS, stationary_patience=STATIONARY_PATIENCE):
    """Transition-matrix ODE kernel: the Euler loop and Stationary Break Check of
    `_simulate_euler_multi`, driven by `_transition_drive`'s sum_a sum_alpha sum_beta
    coupling instead of the fixed successor chain."""
    N = theta_init.shape[0]
    K, P = xi.shape[0], xi.shape[1]

    history = np.empty((num_steps + 1, N), dtype=np.float64)
    theta = theta_init.copy()
    history[0] = theta

    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    m = np.empty((K, P, N), dtype=np.float64)
    V = np.empty(N, dtype=np.float64)

    stationary_count = 0
    actual_steps = num_steps

    for step in range(num_steps):
        for i in range(N):
            cos_theta[i] = np.cos(theta[i])
            sin_theta[i] = np.sin(theta[i])

        _transition_drive(cos_theta, xi, T, d, m, V)

        max_abs_dtheta = 0.0
        for i in range(N):
            dtheta_i = omega[i] - sin_theta[i] * V[i]
            theta[i] = theta[i] + dt * dtheta_i
            if abs(dtheta_i) > max_abs_dtheta:
                max_abs_dtheta = abs(dtheta_i)

        history[step + 1] = theta.copy()

        if max_abs_dtheta < stationary_eps:
            stationary_count += 1
            if stationary_count >= stationary_patience:
                actual_steps = step + 1
                break
        else:
            stationary_count = 0

    return history[:actual_steps + 1]


@njit(cache=True, fastmath=True)
def _simulate_maruyama_transition(theta_init, omega, xi, T, d, dt, num_steps, phase_noise, seed,
                                   stationary_eps=STATIONARY_EPS, stationary_patience=STATIONARY_PATIENCE):
    """Transition-matrix SDE kernel: same `_transition_drive` coupling as
    `_simulate_euler_transition`, with the per-step noise kick of `_simulate_maruyama`."""
    np.random.seed(seed)

    N = theta_init.shape[0]
    K, P = xi.shape[0], xi.shape[1]

    history = np.empty((num_steps + 1, N), dtype=np.float64)
    theta = theta_init.copy()
    history[0] = theta

    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    m = np.empty((K, P, N), dtype=np.float64)
    V = np.empty(N, dtype=np.float64)

    noise_scale = phase_noise * np.sqrt(dt)

    stationary_count = 0
    actual_steps = num_steps

    for step in range(num_steps):
        for i in range(N):
            cos_theta[i] = np.cos(theta[i])
            sin_theta[i] = np.sin(theta[i])

        _transition_drive(cos_theta, xi, T, d, m, V)

        max_abs_dtheta = 0.0
        for i in range(N):
            dtheta_i = omega[i] - sin_theta[i] * V[i]
            random_shock = np.random.randn()
            theta[i] = theta[i] + (dt * dtheta_i) + (noise_scale * random_shock)
            if abs(dtheta_i) > max_abs_dtheta:
                max_abs_dtheta = abs(dtheta_i)

        history[step + 1] = theta.copy()

        # Same stationarity signal as the ODE case: the deterministic drift alone
        # having settled means further integration is just wandering near a fixed point.
        if max_abs_dtheta < stationary_eps:
            stationary_count += 1
            if stationary_count >= stationary_patience:
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

    The only thing that changes is the coupling, so the only thing overridden is
    `_integrate`, the one method that turns patterns into a trajectory. `simulate` and
    every diagnostic (`overlaps`, both decoders, `compare_to_stored_sequence`, `plot`)
    are inherited verbatim and behave identically -- they read `sequences.xi()` and
    `sequences.labels()`, neither of which this module touches.

    T is fixed per network, like every other run parameter: sweeping it across trials
    means constructing a new `TransitionNetwork` per matrix while reusing the same
    CONFIG, exactly as the base module sweeps N or seed.

    Note that `simulate`'s inherited `boundary` argument belongs to the `xi_next`
    dynamics and is rejected here -- with a transition matrix, what the last pattern
    points at is column P-1 of T (see `shift_transition`).
    """

    def __init__(self, transition, **kwargs):
        """`transition` is the (K, P, P) tensor, or anything `_resolve_transition`
        accepts: a single (P, P) matrix shared by every sequence, a builder called with
        P, or a per-sequence list. It is resolved lazily, at `simulate` time, since K and
        P are properties of the sequences being driven, not of the network. Remaining
        keyword arguments are the base `KuramotoNetwork` config."""
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
        """Same contract as `KuramotoNetwork._integrate` -- resolve per-call overrides,
        run the requested kernel, return (theta_history, time) -- over the transition
        kernels above.

        A lone `Sequence` is promoted to the K = 1 case of the (K, P, N) tensor rather
        than given its own kernel: the sum over `a` is then a one-iteration loop, which
        costs nothing and keeps one implementation of the rule.

        `T` here is the base class's simulation *duration*, not the transition matrix
        (which is `self.transition`) -- the name is inherited."""
        if boundary is not None:
            raise ValueError(
                "boundary belongs to the xi_next dynamics; with a transition matrix, what "
                "the last pattern points at is column P-1 of T (see shift_transition)."
            )

        mode = _default(mode, self.mode)
        seed = _default(seed, self.seed)
        T = _default(T, self.T)
        num_steps = self.num_steps_for(T)

        if isinstance(sequences, MultiSequence):
            xi = sequences.xi_tensor()
        else:
            xi = sequences.xi()[np.newaxis]     # K = 1
        xi = np.ascontiguousarray(xi)
        transition = _resolve_transition(self.transition, xi.shape[0], xi.shape[1])

        if mode == "ode":
            theta_history = _simulate_euler_transition(
                theta_init, self.omega, xi, transition, self.d, self.dt, num_steps,
            )
        elif mode == "sde":
            theta_history = _simulate_maruyama_transition(
                theta_init, self.omega, xi, transition, self.d, self.dt, num_steps,
                self.phase_noise, seed,
            )
        else:
            raise ValueError("mode must be either 'ode' or 'sde'")

        actual_steps = theta_history.shape[0] - 1
        time = np.linspace(0.0, actual_steps * self.dt, actual_steps + 1)
        return theta_history, time


# ---------------------------------------------------------------------------
# Example usage (only runs when executed directly, not on import)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import kuramoto_library2 as kl

    CONFIG = dict(dt=0.02, T=55.0, frequency_std=0.03, phase_noise=0.01,
                  d=3, corruption_rate=0.2, tolerance=0.4, mode="ode")

    multi = kl.generate_multi_sequence(N=40, seq_len=5, corruption_rate=0.2, count=2, seed=10)
    P = len(multi[0])

    # gamma * I + Sigma, cycling: each sequence dwells on a pattern, advances, and loops.
    transition = mixed_transition(P, gamma=0.3, kappa=1.0, boundary="cycle")

    net = TransitionNetwork(transition=transition, N=40, seed=10, **CONFIG)
    result = net.simulate(multi)
    print(net.decode_sequence_peaks(result, multi, min_distance=50))
    net.plot(result, multi)
