"""
Lag-Context Extension of the High-Order Kuramoto Associative Memory
====================================================================

kuramoto_context.py
│
├─ [1-95]    docstring + imports        ── 6 names from kuramoto_library2
│
├─ [106-252] BUILDERS  (pure numpy, no state)
│              term(target, lags, weight)          → (1, d+2)
│              self_rule / forward_rule / skip_rule → (1, d+2)
│              bigram_rule(lag)                     → (1, d+2)
│              multilag_rule(weights)               → (n, d+2)
│              coherent_bigram_rule(lag)            → (1, d+2)
│              coherent_trigram_rule(lag)           → (1, d+2)
│              unit_gain(rule, corruption_rate)     → (n, d+2)
│              mixture(*rules)                      → (n, d+2)
│              expand(rule, P, boundary)            → (rows, weights)
│
├─ [255-348] KERNELS  (@njit, no self, no Python objects)
│              _context_drive(cos_theta, xi, rows, weights, m, V)  → fills V
│              _simulate_context(...)                              → history
│
└─ [351-414] NETWORK
               class ContextNetwork(KuramotoNetwork)
                 __init__     stores self.rule
                 _integrate   THE ONLY OVERRIDE



`kuramoto_transition` widened "mu drives mu+1" into a transition matrix T, but its gate

    sum_beta T_{alpha beta} (m^beta)^d

is a sum of *pure powers*: every overlap sits in its own term, and no term multiplies two
different overlaps together. The context rules of the write-up are exactly the terms that
sum cannot reach -- the d factors of the gate are read at *different lags*:

    dtheta_i/dt = omega_i - sin(theta_i) * sum_a sum_mu sum_terms w
                            xi_i^{a, mu+s} * prod_{l in lags} m_i^{a, mu-l}

    m_i^{a,beta} = (1/(N-1)) sum_{j!=i} xi_j^{a,beta} cos(theta_j)

so a rule is a table of terms, one row each: a target offset `s`, `d` source lags, and a
weight. Stored patterns are real (xi = cos(phase) in {-1,+1}), so every overlap is real
and |m|^2 = m^2: the write-up's fourth-order bidegree-(2,1) fields are the d = 3 rows here.

    rule                     eq   term(s, lags)          gate at pattern mu
    ---------------------------------------------------------------------------------
    self_rule               (11)  (0, (0,0,0))           xi^mu     m_mu^3
    forward_rule            (12)  (1, (0,0,0))           xi^{mu+1} m_mu^3
    skip_rule(nhop=2)       (13)  (2, (0,0,0))           xi^{mu+2} m_mu^3
    bigram_rule             (16)  (1, (0,1,1))           xi^{mu+1} m_mu |m_{mu-1}|^2
    multilag_rule           (17)  (1, (0,k,k)) summed    xi^{mu+1} m_mu sum_k w_k |m_{mu-k}|^2
    coherent_bigram_rule    (18)  (1, (0,0,1))           xi^{mu+1} m_mu^2 m_{mu-1}
    coherent_trigram_rule   (19)  (1, (0,1,2))           xi^{mu+1} m_mu m_{mu-1} m_{mu-2}
    mixture(a*A, b*B, ...)  (14)  rows stacked           the linear mixture of its parts

`forward_rule()` recovers the `xi_next` dynamics exactly, so it is the baseline. The two
rules of the write-up that are *not* here are the residual bigram (21) and the ridge
control (22): both replace the target pattern xi^{mu+1} by a fitted or branch-averaged
vector rather than by another stored pattern, so neither is a table of relative indices.

Everything else is `kuramoto_library2`: storage, generation, diagnostics, decoders and
plotting are inherited unchanged. A rule is *relative*, so unlike a transition matrix it
does not have the end of the chain baked in and the inherited `boundary` argument still
means what it always did -- "self" (default) clamps both ends, "cycle" wraps both.

    import kuramoto_library2 as kl
    from kuramoto_context import ContextNetwork, coherent_trigram_rule

    CONFIG = dict(dt=0.02, T=55.0, frequency_mean=0.0, frequency_std=0.03, phase_noise=0.01,
                  d=3, corruption_rate=0.2, tolerance=0.4, mode="ode")
    multi = kl.generate_multi_sequence(N=40, seq_len=5, corruption_rate=0.2, count=2, seed=10)
    net = ContextNetwork(rule=coherent_trigram_rule(), N=40, seed=10, **CONFIG)
    net.plot(net.simulate(multi), multi)
"""

from __future__ import annotations

from typing import Optional, Sequence as TypingSequence, Tuple

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
# Rules -- free functions, like the matrix builders in kuramoto_transition. Each
# returns a plain (n_terms, d+2) array whose columns are
#
#     [ target offset s | lag_1 ... lag_d | weight ]
#
# so a rule is combined with `mixture(...)` (a vstack) and reweighted by passing a
# different `weight`. All lags are >= 0: they count *backwards* from the driving
# pattern mu, so lag 0 is mu itself and lag 1 the pattern before it.
# ---------------------------------------------------------------------------

def term(target: int, lags: TypingSequence[int], weight: float = 1.0) -> np.ndarray:
    """One row: drive xi^{mu+target} with weight * prod_l m^{mu-l}, l over `lags`.

    The general case -- every builder below is one call to this with the lags spelled
    out, and any rule of the family that is not named below is written with it directly.
    """
    return np.array([[float(target), *map(float, lags), float(weight)]])


def self_rule(weight: float = 1.0, d: int = 3) -> np.ndarray:
    """xi^mu m_mu^d -- the autoassociative stabilizer, eq (11).

    Each stored pattern drives itself, which makes it something to sit in but never
    creates a sequence; the natural stabilizing term to add to a chain.
    """
    return term(0, (0,) * d, weight)


def forward_rule(weight: float = 1.0, d: int = 3) -> np.ndarray:
    """xi^{mu+1} m_mu^d -- the Markov-1 rule, eq (12).

    The `xi_next` dynamics of `kuramoto_library2` written as a rule, and the baseline
    every context rule is measured against.
    """
    return term(1, (0,) * d, weight)


def skip_rule(nhop: int = 2, weight: float = 1.0, d: int = 3) -> np.ndarray:
    """xi^{mu+nhop} m_mu^d -- the skip-`nhop` rule, eq (13) at nhop = 2.

    Still current-state-only: it changes which pattern is driven, not what is read.
    """
    return term(nhop, (0,) * d, weight)


def bigram_rule(lag: int = 1, weight: float = 1.0, d: int = 3) -> np.ndarray:
    """xi^{mu+1} m_mu |m_{mu-lag}|^{d-1} -- the magnitude-gated bigram rule, eq (16).

    The simplest context rule: the *next* pattern is chosen by the current overlap, but
    the term is gated by how much overlap with a pattern `lag` steps back survives. At
    lag = 0 this is `forward_rule` exactly, which is why `multilag_rule` can write its
    whole sum with it.
    """
    return term(1, (0,) + (lag,) * (d - 1), weight)


def multilag_rule(weights: TypingSequence[float] = (1.0, 1.0, 1.0), d: int = 3) -> np.ndarray:
    """xi^{mu+1} m_mu sum_k w_k |m_{mu-k}|^{d-1} -- the multilag context rule, eq (17).

    `weights[k]` is the weight on the lag-k gate, so `weights[0]` is the ungated forward
    term and the rule reaches back as far as the last nonzero entry. Zero weights are
    dropped rather than stored as zero rows, so the reach of a rule is visible in its
    length.
    """
    rows = [bigram_rule(lag=k, weight=w, d=d) for k, w in enumerate(weights) if w]
    return np.vstack(rows)


def coherent_bigram_rule(lag: int = 1, weight: float = 1.0, d: int = 3) -> np.ndarray:
    """xi^{mu+1} m_mu^{d-1} m_{mu-lag} -- the coherent bigram rule, eq (18).

    `bigram_rule` with one factor moved from the gate to the drive: the lagged overlap
    enters linearly rather than squared, so its *sign* matters and not only its size.
    """
    return term(1, (0,) * (d - 1) + (lag,), weight)


def coherent_trigram_rule(lag: int = 2, weight: float = 1.0, d: int = 3) -> np.ndarray:
    """xi^{mu+1} m_mu m_{mu-lag+1} m_{mu-lag} -- the coherent trigram rule, eq (19).

    Two-step context inside the same fourth-order budget: the three factors are read at
    three different lags, so the rule can still tell which branch the trajectory came
    from `lag` patterns ago. `lag` is that reach, and eq (19) is lag = 2.
    """
    return term(1, (0,) * (d - 2) + (lag - 1, lag), weight)


def unit_gain(rule: np.ndarray, corruption_rate: float) -> np.ndarray:
    """Rescale every term so its gate is ~1 when the state sits on the driving pattern.

    The write-up carries a global coupling gain `g`; this is what has to be set for a
    comparison between rules to be about *information* rather than about drive strength.
    A rule that reads only m_mu has a gate of ~1 whenever the state is on pattern mu, but
    one that also reads m_{mu-l} is multiplied there by the residual overlap with a
    pattern l mutations back, which for a corruption chain at rate rho is about
    (1 - 2 rho)^l. So a lag-2 gate runs an order of magnitude weaker than the forward
    rule's and, un-normalized, simply fails to traverse -- which says nothing about
    whether the context it reads is useful.

    Dividing each term by (1 - 2 rho)^{sum of its lags} removes exactly that handicap, and
    dividing the rule by its total weight then puts every rule at the same total gate,
    however many terms it has. Both are per-term constants, so nothing changes but the
    scale of the drive; this is the counterpart of `lag_matrix`'s normalization of a
    transition matrix to unit total weight, and it is what makes a mixture's weights mean
    "relative contribution at the operating point".
    """
    rule = np.atleast_2d(np.asarray(rule, dtype=np.float64)).copy()
    rule[:, -1] /= rule[:, -1].sum()                                        # weights -> shares
    rule[:, -1] /= (1.0 - 2.0 * corruption_rate) ** rule[:, 1:-1].sum(axis=1)
    return rule


def mixture(*rules: np.ndarray) -> np.ndarray:
    """Linear mixture of rules, eq (14) -- stack their rows.

    The weights live in the rows themselves, so `mixture(self_rule(0.3), forward_rule())`
    is the mixture with a_0 = 0.3 and a_1 = 1.
    """
    return np.vstack(rules)


def expand(rule: np.ndarray, P: int, boundary: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Relative rule -> absolute pattern indices, one row per (term, mu) pair.

    Returns `(rows, weights)`: `rows[r]` is `[alpha, beta_1, ... beta_d]`, the pattern
    driven and the d patterns read, and `weights[r]` the coefficient. Both ends of the
    chain are resolved the same way, by the inherited `boundary`:

    - "self" (default) clamps, so mu + s past the end is the terminal pattern and
      mu - lag before the start is the first one. Clamping at the *start* is what lets a
      cue at pattern 0 launch the sequence at all: a lag-1 gate there has no earlier
      overlap to read, and reading m_0 twice degrades the term to `forward_rule`.
    - "cycle" wraps both, mod P, which is the consistent reading for a looping sequence:
      the pattern before the first is the last.
    """
    boundary = _default(boundary, "self")
    if boundary == "self":
        resolve = lambda idx: np.clip(idx, 0, P - 1)
    elif boundary == "cycle":
        resolve = lambda idx: idx % P
    else:
        raise ValueError("boundary must be 'self' or 'cycle'")

    rule = np.atleast_2d(np.asarray(rule, dtype=np.float64))
    offsets, lags, weights = rule[:, 0], rule[:, 1:-1], rule[:, -1]

    mu = np.arange(P)[:, None]                              # (P, 1), broadcast over terms
    alpha = resolve(mu + offsets[None, :].astype(int))      # (P, n_terms)
    beta = resolve(mu[:, :, None] - lags[None, :, :].astype(int))   # (P, n_terms, d)

    rows = np.concatenate([alpha[:, :, None], beta], axis=2).reshape(-1, lags.shape[1] + 1)
    return np.ascontiguousarray(rows.astype(np.int64)), \
           np.ascontiguousarray(np.tile(weights, P))


# ---------------------------------------------------------------------------
# Low-level numba kernels (free functions -- numba methods can't take `self`)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _context_drive(cos_theta, xi, rows, weights, m, V):
    """V[i] = sum_a sum_r w_r xi_i^{a, rows[r,0]} prod_{k=1}^{d} m_i^{a, rows[r,k]}.

    `_transition_drive` with the single sharpened overlap `h` replaced by a product over
    the term's d source patterns. Because those sources differ from factor to factor, the
    overlaps can no longer be sharpened as they are computed and thrown away: all K*P of
    them are materialized into `m` first, then read by index. That is the whole cost of
    lag context -- one (K, P, N) scratch buffer, and the same power count as before.
    """
    K, P, N = xi.shape
    d = rows.shape[1] - 1

    for i in range(N):
        V[i] = 0.0

    for a in range(K):
        for beta in range(P):
            overlap = 0.0
            for j in range(N):
                overlap += xi[a, beta, j] * cos_theta[j]
            for i in range(N):
                # drop the j == i term, exactly as the base kernels do
                m[a, beta, i] = (overlap - xi[a, beta, i] * cos_theta[i]) / (N - 1)

    for a in range(K):
        for r in range(rows.shape[0]):
            weight = weights[r]
            if weight == 0.0:
                continue
            alpha = rows[r, 0]
            for i in range(N):
                gate = weight
                for k in range(1, d + 1):
                    gate *= m[a, rows[r, k], i]
                V[i] += xi[a, alpha, i] * gate


@njit(cache=True, fastmath=True)
def _simulate_context(theta_init, omega, xi, rows, weights, dt, num_steps, phase_noise, seed):
    """Euler/Euler-Maruyama loop and Stationary Break Check of `_simulate_transition`,
    driven by `_context_drive`. phase_noise = 0 is the ODE, > 0 the SDE.

    Stationarity is judged on the deterministic drift alone in both cases: once that has
    settled, further integration is just wandering near a fixed point.
    """
    np.random.seed(seed)
    N = theta_init.shape[0]
    K, P = xi.shape[0], xi.shape[1]

    history = np.empty((num_steps + 1, N), dtype=np.float64)
    theta = theta_init.copy()
    history[0] = theta

    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    V = np.empty(N, dtype=np.float64)
    m = np.empty((K, P, N), dtype=np.float64)     # the overlap scratch buffer

    noise_scale = phase_noise * np.sqrt(dt)
    stationary_count = 0
    actual_steps = num_steps

    for step in range(num_steps):
        for i in range(N):
            cos_theta[i] = np.cos(theta[i])
            sin_theta[i] = np.sin(theta[i])

        _context_drive(cos_theta, xi, rows, weights, m, V)

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
# Network: KuramotoNetwork with the lag-context coupling swapped in
# ---------------------------------------------------------------------------

class ContextNetwork(KuramotoNetwork):
    """`KuramotoNetwork` driven by a lag-context rule instead of the successor chain.

    Only the coupling changes, so `_integrate` is the only override: `simulate` and every
    diagnostic (`overlaps`, both decoders, `compare_to_stored_sequence`, `plot`) are
    inherited verbatim. The rule is fixed per network, like every other run parameter, so
    sweeping it means constructing a new `ContextNetwork` per rule.
    """

    def __init__(self, rule, **kwargs):
        """`rule` is an (n_terms, d+2) table from the builders above, shared by every
        stored sequence. Remaining kwargs are the base config."""
        super().__init__(**kwargs)
        self.rule = rule

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

        `T` here is the inherited *duration* argument, not a transition matrix.
        """
        mode = _default(mode, self.mode)
        num_steps = self.num_steps_for(_default(T, self.T))

        xi = sequences.xi_tensor() if isinstance(sequences, MultiSequence) else sequences.xi()[np.newaxis]
        xi = np.ascontiguousarray(xi)
        P = xi.shape[1]

        rows, weights = expand(self.rule, P, boundary)
        if rows.shape[1] - 1 != self.d:
            raise ValueError(
                f"Rule has {rows.shape[1] - 1} overlap factors per term but the network runs "
                f"at d = {self.d}; pass d={rows.shape[1] - 1} in the config, or build the rule "
                f"with d={self.d}."
            )

        if mode not in ("ode", "sde"):
            raise ValueError("mode must be either 'ode' or 'sde'")

        theta_history = _simulate_context(
            theta_init, self.omega, xi, rows, weights, self.dt, num_steps,
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

    # The coherent trigram: the next pattern is driven by the current overlap, gated by
    # the two before it, so the rule still knows where the trajectory came from.
    net = ContextNetwork(rule=coherent_trigram_rule(), N=40, seed=10, **CONFIG)
    result = net.simulate(multi)
    print(net.decode_sequence_peaks(result, multi, min_distance=50))
    net.plot(result, multi)
