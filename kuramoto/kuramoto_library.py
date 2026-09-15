"""
High-Order Kuramoto Associative Memory Network (OOP)
=====================================================

Model
-----
Single sequence (period P, boundary condition xi^{P+1} == ?):

    dtheta_i/dt = omega_i - sin(theta_i) * sum_mu xi_i^{mu+1} * ( (1/(N-1)) sum_{j!=i} xi_j^mu cos(theta_j) )^d

`omega_i` defaults to 0 (no intrinsic-frequency heterogeneity) -- see `KuramotoNetwork`.
What xi^{P+1} means is a choice, not fixed: None/"self" (default) sets xi^{P+1} == xi^P,
the last pattern pointing to itself so the dynamics settle there instead of cycling
indefinitely; "cycle" sets xi^{P+1} == xi^1, wrapping back to the first pattern. See
`Sequence.xi_next`.

The sum over mu is implemented directly as a "successor pattern" tensor `xi_next`, the
same shape as `xi`: row mu of `xi_next` is xi^{mu+1} (row mu+1 of `xi`) for every row but
the last, whose successor is either itself or row 0 depending on the boundary condition
above. No separate edge-index arrays are needed -- since a pattern's identity is just
its row position (see below), "which pattern comes after mu" is exactly the
shift-by-one-and-wrap that produces `xi_next`, which can be sliced out directly rather
than reconstructed as an index array and re-dereferenced.

Storage vs. computation
------------------------
Three data classes, no bookkeeping between them:

- `VectorSequence`   : list-like indexing (`A[i]`) over a contiguous (L, N) array.
- `Sequence`         : one named `VectorSequence` of phase vectors (theta in {0, pi}^N).
                       A pattern's identity *is* its row position (`seq[k]`), so there's
                       no separate name -> index table to keep in sync.
- `MultiSequence`    : an ordered collection of named `Sequence`s, exposed as one 3D
                       tensor `M` (`M[0]` is the first sequence added, e.g. "A") when
                       every member shares a length, or indexed individually otherwise.

`Sequence` and `MultiSequence` both expose the same trio of read-only views --
`.xi()`, `.xi_next()`, `.labels()` -- so the diagnostics (`overlaps`, both decoders,
`plot`) take either one without knowing which they hold.

For integration, `MultiSequence` additionally exposes the rectangular (K, P, N) pair
`.xi_tensor()` / `.xi_next_tensor(boundary)`, putting the sequence index `a` on its own
axis (which is why its members must then share a length P). `KuramotoNetwork._integrate`
picks that pair for a `MultiSequence` and the plain (P, N) `.xi()` / `.xi_next(boundary)`
for a lone `Sequence`, then runs the matching kernel -- 2D is the single-sequence sum
over mu, 3D the multi-sequence sum over (a, mu). Kernel choice stays out of the data
classes, which only know how to store and describe vectors.

Pattern generation/mutation is *not* a method on either data class -- it lives in free
functions (`generate_sequence`, `generate_multi_sequence`), the same way the ODE/SDE
integration lives in free kernel functions (`_simulate_euler`, `_simulate_maruyama`)
rather than as network methods. The data classes only know how to store and describe
vectors; only the generation functions and the kernels know how to produce/evolve them.

Typical usage
-------------
    CONFIG = dict(dt=0.02, T=55.0, frequency_mean=0.0, frequency_std=0.03, phase_noise=0.01,
                  d=3, corruption_rate=0.2, tolerance=0.4, mode="ode")
    net = KuramotoNetwork(N=40, seed=10, **CONFIG)
    A = generate_sequence(N=40, seq_len=5, corruption_rate=CONFIG["corruption_rate"], name="A", seed=10)
    result = net.simulate(A)
    net.plot(result, A)

Sweeping N or seed across trials just means constructing a new `KuramotoNetwork` per
value while reusing the same `CONFIG`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from typing import Sequence as TypingSequence

import numpy as np

from scipy.signal import find_peaks

try:
    from numba import njit
except ImportError:  # pragma: no cover - allows the module to still import without numba
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrap(fn):
            return fn
        return wrap


def _default(value, fallback):
    """Return `value` unless it's None, in which case return `fallback`."""
    return fallback if value is None else value


# ---------------------------------------------------------------------------
# Storage: one ordered sequence of N-dimensional vectors
# ---------------------------------------------------------------------------

class VectorSequence:
    """Encapsulates an ordered sequence of N-dimensional vectors.

    Combines clean list-like indexing (`A[i]`) with contiguous 2D array performance.
    Pre-allocate with `L` and `N` known (fastest: fixed-size mutation chains); leave them
    out to build vector-by-vector with `append` when the final length isn't known yet.
    """

    def __init__(self, L: Optional[int] = None, N: Optional[int] = None):
        self.L = L
        self.N = N

        if L is not None and N is not None:
            self._matrix = np.zeros((L, N))
            self._is_preallocated = True
        else:
            self._data_list: List[np.ndarray] = []
            self._is_preallocated = False

    def append(self, vector) -> None:
        """Appends a new vector if sequence is dynamically sized."""
        if self._is_preallocated:
            raise RuntimeError("Cannot append to a pre-allocated sequence. Use indexed assignment (e.g., A[i] = v).")

        vec = np.asarray(vector)
        if self.N is None:
            self.N = vec.shape[0]
        elif vec.shape[0] != self.N:
            raise ValueError(f"Dimension mismatch: expected {self.N}, got {vec.shape[0]}")

        self._data_list.append(vec)

    def as_matrix(self) -> np.ndarray:
        """Exposes the underlying L x N array for fast bulk linear algebra."""
        if self._is_preallocated:
            return self._matrix
        return np.stack(self._data_list) if self._data_list else np.empty((0, self.N or 0))

    # --- Python Magic Methods ---

    def __getitem__(self, idx):
        """Allows direct indexing: A[0], A[1], or slicing A[1:3]."""
        if self._is_preallocated:
            return self._matrix[idx]
        return self._data_list[idx]

    def __setitem__(self, idx, value) -> None:
        """Allows direct assignment: A[0] = vector."""
        if self._is_preallocated:
            self._matrix[idx] = value
        else:
            self._data_list[idx] = np.asarray(value)

    def __len__(self) -> int:
        return self.L if self._is_preallocated else len(self._data_list)

    def __repr__(self) -> str:
        mode = "Pre-allocated" if self._is_preallocated else "Dynamic"
        return f"<VectorSequence [{mode}] | Length: {len(self)}, Dim: {self.N}>"


# ---------------------------------------------------------------------------
# Sequence / MultiSequence: named storage + the read-only views the network needs
# ---------------------------------------------------------------------------

class Sequence:
    """One named, ordered chain of phase-vector patterns (theta in {0, pi}^N).

    A thin wrapper around a `VectorSequence` that additionally knows its own name and
    can describe itself to `KuramotoNetwork`: its spin representation (`xi`), its
    successor-pattern tensor (`xi_next`, encoding the mu -> mu+1 chain with a choice of
    boundary condition at the end -- see the module docstring), and a cue drawn from
    one of its own patterns. Bulk generation lives in `generate_sequence` below;
    `append` here only grows one sequence by one pattern at a time.
    """

    def __init__(self, name: str, vectors: VectorSequence):
        self.name = name
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]

    def as_matrix(self) -> np.ndarray:
        return self.vectors.as_matrix()

    def xi(self) -> np.ndarray:
        """Spin representation (+-1) of every pattern in this sequence: cos(phase)."""
        return np.cos(self.as_matrix())

    def xi_hop(self, boundary: Optional[str] = None, nhop: int = 1) -> np.ndarray:
        """Same shape as `xi()`: row mu is the pattern that mu points *nhop steps ahead*
        to -- row mu+nhop of `xi()`, with rows that would run off the end resolved by
        `boundary`:

        - None (default) or "self": clamped to the last row, xi^P. Every pattern within
          nhop of the end points at the terminal pattern, so the dynamics settle there.
          (Symmetrically, a negative `nhop` clamps at row 0.)
        - "cycle": taken mod P, so mu -> (mu + nhop) % P -- periodic wraparound.

        `nhop=1` is the ordinary successor chain and is exactly what `xi_next` returns.
        Larger `nhop` skips intermediate patterns (mu -> mu+2 -> mu+4 ... for nhop=2);
        `nhop=0` makes every pattern its own successor, freezing the chain; a negative
        `nhop` runs it backwards.

        Implemented as a gather through a row-index array rather than a slice-and-stack,
        since for nhop > 1 the clamped case is no longer a single contiguous slice plus
        one repeated row.
        """
        boundary = _default(boundary, "self")
        xi = self.xi()
        rows = np.arange(xi.shape[0])
        if boundary == "self":
            rows = np.clip(rows + nhop, 0, xi.shape[0] - 1)
        elif boundary == "cycle":
            rows = (rows + nhop) % xi.shape[0]
        else:
            raise ValueError("boundary must be 'self' or 'cycle'")
        return xi[rows]

    def xi_next(self, boundary: Optional[str] = None) -> np.ndarray:
        """Same shape as `xi()`: row mu is the pattern mu transitions *to* -- row mu+1
        of `xi()` for every row but the last, whose successor depends on `boundary`:

        - None (default) or "self": itself, xi^P -- the terminal pattern, so the
          dynamics settle there instead of cycling back to the first.
        - "cycle": row 0, xi^1 -- periodic wraparound, xi^{P+1} == xi^1.

        The single-step case of `xi_hop`, which this delegates to.
        """
        return self.xi_hop(boundary=boundary, nhop=1)

    def labels(self) -> List[Tuple[str, int]]:
        """(name, position) label for every pattern, in `xi()`/`xi_next()` row order."""
        return [(self.name, idx) for idx in range(len(self))]

    def cue(self, idx: int = 0, corruption_rate: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
        """Corrupted copy of pattern `idx`, for use as an initial condition theta(0)."""
        return corrupt_phase(self[idx], corruption_rate, np.random.default_rng(seed))

    def append(
        self,
        vector: Optional[np.ndarray] = None,
        corruption_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Grow this sequence by one pattern (only valid on a dynamically-sized
        `Sequence`, i.e. one built on a `VectorSequence` with no `L` -- see
        `VectorSequence.append`).

        Polymorphic: pass an explicit `vector` to append it as-is. Omit it and the
        next pattern is generated automatically instead -- a fresh random pattern if
        this is the first one (xi^1), otherwise a corrupted copy of the current last
        pattern (xi^{k+1} = mutate(xi^k, corruption_rate)), the same rule
        `generate_sequence` uses to build a whole chain up front. Returns the
        (possibly auto-generated) vector that was appended."""
        rng = np.random.default_rng(seed)
        if vector is None:
            if len(self) == 0:
                if self.vectors.N is None:
                    raise ValueError(
                        "Cannot auto-generate the first pattern without a known N; "
                        "construct with VectorSequence(N=N) or pass an explicit vector."
                    )
                vector = rng.choice([0.0, np.pi], size=self.vectors.N)
            else:
                if corruption_rate is None:
                    raise ValueError("corruption_rate is required to mutate the previous pattern.")
                vector = corrupt_phase(self[-1], corruption_rate, rng)
        self.vectors.append(vector)
        return vector

    def __repr__(self) -> str:
        return f"<Sequence '{self.name}' | {self.vectors!r}>"


class MultiSequence:
    """An ordered collection of named `Sequence`s.

    `add`-order is preserved as the index into the stacked tensor `M`, so if sequences
    "A" and "B" were added in that order, `M[0]` is "A"'s (L, N) matrix and `M[1]` is
    "B"'s. Exposes the same `.xi()` / `.xi_next()` / `.labels()` views as a lone
    `Sequence` (each optionally restricted to a subset of names), so `KuramotoNetwork`
    drives a `MultiSequence` exactly like it drives a `Sequence` -- a single sequence is
    just the one-member case of this.
    """

    _AUTO_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self):
        self._by_name: Dict[str, Sequence] = {}
        self._order: List[str] = []

    def add(self, sequence: Sequence) -> Sequence:
        if sequence.name in self._by_name:
            raise ValueError(f"Sequence name already exists: {sequence.name}")
        self._by_name[sequence.name] = sequence
        self._order.append(sequence.name)
        return sequence

    def auto_name(self) -> str:
        idx = len(self._order)
        if idx < len(self._AUTO_NAMES):
            return self._AUTO_NAMES[idx]
        return f"S{idx}"

    @property
    def names(self) -> List[str]:
        return list(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self):
        return (self._by_name[name] for name in self._order)

    def __getitem__(self, key) -> Sequence:
        """`multi["A"]` by name, or `multi[0]` by add-order position."""
        if isinstance(key, str):
            return self._by_name[key]
        return self._by_name[self._order[key]]

    @property
    def M(self) -> np.ndarray:
        """3D tensor stacking every member's phase matrix in add-order: M[a] is the
        a-th sequence added (requires every member to share the same length --
        mixed-length sequences can only be accessed individually via `multi[name]`)."""
        if not self._order:
            return np.empty((0, 0, 0))
        return np.stack([self._by_name[name].as_matrix() for name in self._order])

    def _selected(self, names: Optional[TypingSequence[str]] = None) -> List[Sequence]:
        names = list(names) if names is not None else self._order
        return [self._by_name[name] for name in names]

    def xi(self, names: Optional[TypingSequence[str]] = None) -> np.ndarray:
        return np.concatenate([seq.xi() for seq in self._selected(names)], axis=0)

    def xi_next(self, names: Optional[TypingSequence[str]] = None, boundary: Optional[str] = None) -> np.ndarray:
        """Concatenation of each selected sequence's own `xi_next(boundary)` -- each
        sequence wraps (or self-loops) independently of the others, per its own
        `xi^{a,P_a+1} == xi^{a,1}` ("cycle") or `== xi^{a,P_a}` (None/"self"), so this
        needs no offset arithmetic: it's just each member's local shift-and-wrap,
        stacked in the same order as `xi(names)`."""
        return np.concatenate([seq.xi_next(boundary=boundary) for seq in self._selected(names)], axis=0)

    def labels(self, names: Optional[TypingSequence[str]] = None) -> List[Tuple[str, int]]:
        return [label for seq in self._selected(names) for label in seq.labels()]

    def _uniform_selected(self, names: Optional[TypingSequence[str]] = None) -> List[Sequence]:
        """The selected sequences, requiring every one to store the same number of
        patterns P -- the precondition for the rectangular (K, P, N) views below."""
        seqs = self._selected(names)
        if not seqs:
            raise ValueError("No sequences selected.")
        if len({len(seq) for seq in seqs}) != 1:
            raise ValueError(
                f"Every stored sequence must have the same length, got "
                f"{ {seq.name: len(seq) for seq in seqs} }."
            )
        return seqs

    def xi_tensor(self, names: Optional[TypingSequence[str]] = None) -> np.ndarray:
        """Order-3 spin tensor, shape (K, P, N): `xi_tensor()[a, mu, i]` is neuron i of
        pattern mu of sequence a. The sequence index `a` is an axis of the array, which
        is what lets the kernels below write `sum_{a=1}^{K} sum_{mu=1}^{P}` as a plain
        nested loop over the first two axes."""
        return np.stack([seq.xi() for seq in self._uniform_selected(names)])

    def xi_next_tensor(self, names: Optional[TypingSequence[str]] = None,
                       boundary: Optional[str] = None) -> np.ndarray:
        """Successor tensor matching `xi_tensor()`, shape (K, P, N):
        `xi_next_tensor()[a, mu]` is what pattern mu of sequence a transitions *to*.

        Each sequence's terminal pattern is resolved by that sequence's own boundary
        condition, applied along its own `mu` axis before stacking -- so with the
        default None/"self" the last slice self-references (`xi_next[a, P-1] ==
        xi[a, P-1]`, the sequence settles there), and with "cycle" it wraps to that
        same sequence's first pattern (`xi_next[a, P-1] == xi[a, 0]`). A successor is
        never taken from a different `a`."""
        return np.stack([seq.xi_next(boundary=boundary) for seq in self._uniform_selected(names)])

    def __repr__(self) -> str:
        return f"<MultiSequence | {self._order}>"


# ---------------------------------------------------------------------------
# Generation / mutation -- free functions, not methods (mirrors the kernels below)
# ---------------------------------------------------------------------------

def corrupt_phase(phase: np.ndarray, corruption_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Random subset of neurons flipped 0 <-> pi. Pure: takes and returns a copy,
    draws from the given `rng` rather than owning one."""
    phase = phase.copy()
    if corruption_rate <= 0:
        return phase
    num_flips = int(corruption_rate * phase.shape[0])
    flip_idx = rng.choice(phase.shape[0], size=num_flips, replace=False)
    phase[flip_idx] = np.pi - phase[flip_idx]
    return phase


def sequence_from_patterns(name: str, patterns: TypingSequence[np.ndarray]) -> Sequence:
    """Build a `Sequence` directly from an explicit, already-computed list of phase
    vectors -- the manual-construction counterpart to `generate_sequence`'s
    auto-mutated chain, for when you need specific patterns in specific positions
    (e.g. reusing the same pattern across multiple sequences, as in a
    crossover/intersection experiment)."""
    N = patterns[0].shape[0]
    vectors = VectorSequence(L=len(patterns), N=N)
    for i, pattern in enumerate(patterns):
        vectors[i] = pattern
    return Sequence(name=name, vectors=vectors)


def generate_sequence(
    N: int,
    seq_len: int,
    corruption_rate: float,
    name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Sequence:
    """Build one length-(seq_len+1) `Sequence`: a fresh random first pattern xi^1, then
    seq_len corruption steps (xi^{k+1} = mutate(xi^k, corruption_rate)). Auto-named "A"
    if `name` is omitted."""
    rng = np.random.default_rng(seed)
    vectors = VectorSequence(L=seq_len + 1, N=N)
    vectors[0] = rng.choice([0.0, np.pi], size=N)
    for k in range(1, seq_len + 1):
        vectors[k] = corrupt_phase(vectors[k - 1], corruption_rate, rng)
    return Sequence(name=_default(name, "A"), vectors=vectors)


def generate_multi_sequence(
    N: int,
    seq_len: int,
    corruption_rate: float,
    names: Optional[TypingSequence[Optional[str]]] = None,
    count: Optional[int] = None,
    seed: Optional[int] = None,
) -> MultiSequence:
    """Generate several independent `Sequence`s at once (each its own fresh random
    first pattern + seq_len mutation steps), collected into one `MultiSequence`. Pass
    `names` explicitly, or `count` to let them auto-name ("A", "B", ...)."""
    rng = np.random.default_rng(seed)
    names = _default(names, [None] * _default(count, 1))

    multi = MultiSequence()
    for name in names:
        name = _default(name, multi.auto_name())
        seq = generate_sequence(N, seq_len, corruption_rate, name=name, seed=int(rng.integers(0, 2**32 - 1)))
        multi.add(seq)
    return multi


# ---------------------------------------------------------------------------
# Low-level numba kernels (free functions -- numba methods can't take `self`)
# ---------------------------------------------------------------------------

# Stationary Break Check: if every neuron's own dtheta/dt stays below STATIONARY_EPS
# for STATIONARY_PATIENCE consecutive steps, the trajectory has settled and further
# integration is wasted -- stop early rather than running out the full schedule.
STATIONARY_EPS = 1e-6
STATIONARY_PATIENCE = 10


@njit(cache=True, fastmath=True)
def _simulate_euler(theta_init, omega, xi, xi_next, d, dt, num_steps,
                     stationary_eps=STATIONARY_EPS, stationary_patience=STATIONARY_PATIENCE):
    N = theta_init.shape[0]
    num_patterns = xi.shape[0]

    history = np.empty((num_steps + 1, N), dtype=np.float64)
    theta = theta_init.copy()
    history[0] = theta

    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    S = np.empty(num_patterns, dtype=np.float64)

    stationary_count = 0
    actual_steps = num_steps

    for step in range(num_steps):
        for i in range(N):
            cos_theta[i] = np.cos(theta[i])
            sin_theta[i] = np.sin(theta[i])

        # Overlap with every stored pattern, computed once and reused below
        for p in range(num_patterns):
            temp_sum = 0.0
            for j in range(N):
                temp_sum += xi[p, j] * cos_theta[j]
            S[p] = temp_sum

        V = np.zeros(N, dtype=np.float64)
        for p in range(num_patterns):
            for i in range(N):
                inner_sum = S[p] - xi[p, i] * cos_theta[i]
                h = (inner_sum / (N - 1)) ** d
                V[i] += xi_next[p, i] * h

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
def _simulate_maruyama(theta_init, omega, xi, xi_next, d, dt, num_steps, phase_noise, seed,
                        stationary_eps=STATIONARY_EPS, stationary_patience=STATIONARY_PATIENCE):
    np.random.seed(seed) # for @njit, we have to use np.random.seed inside

    N = theta_init.shape[0]
    num_patterns = xi.shape[0]

    history = np.empty((num_steps + 1, N), dtype=np.float64)
    theta = theta_init.copy()
    history[0] = theta

    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    S = np.empty(num_patterns, dtype=np.float64)

    noise_scale = phase_noise * np.sqrt(dt)

    stationary_count = 0
    actual_steps = num_steps

    for step in range(num_steps):
        for i in range(N):
            cos_theta[i] = np.cos(theta[i])
            sin_theta[i] = np.sin(theta[i])

        for p in range(num_patterns):
            temp_sum = 0.0
            for j in range(N):
                temp_sum += xi[p, j] * cos_theta[j]
            S[p] = temp_sum

        V = np.zeros(N, dtype=np.float64)
        for p in range(num_patterns):
            for i in range(N):
                inner_sum = S[p] - xi[p, i] * cos_theta[i]
                h = (inner_sum / (N - 1)) ** d
                V[i] += xi_next[p, i] * h

        max_abs_dtheta = 0.0
        for i in range(N):
            dtheta_i = omega[i] - sin_theta[i] * V[i]
            random_shock = np.random.randn()
            theta[i] = theta[i] + (dt * dtheta_i) + (noise_scale * random_shock)
            if abs(dtheta_i) > max_abs_dtheta:
                max_abs_dtheta = abs(dtheta_i)

        history[step + 1] = theta.copy()

        # Noise keeps the SDE trajectory moving forever in principle, but once the
        # deterministic drift alone (dtheta_i, ignoring the noise kick) has settled
        # for this long, further integration is just wandering near a fixed point --
        # the same stationarity signal as the ODE case.
        if max_abs_dtheta < stationary_eps:
            stationary_count += 1
            if stationary_count >= stationary_patience:
                actual_steps = step + 1
                break
        else:
            stationary_count = 0

    return history[:actual_steps + 1]


# --- Multi-sequence (K stored sequences, P patterns each) --------------------
#
# `xi`/`xi_next` here are the rectangular (K, P, N) tensors from
# `MultiSequence.xi_tensor()` / `.xi_next_tensor()`: the sequence index `a` is axis 0
# and the within-sequence pattern index `mu` is axis 1, so the two-index sum below is
# just a nested loop over those two axes. Every stored sequence must have the same
# length P (`MultiSequence` raises otherwise).

@njit(cache=True, fastmath=True)
def _multi_drive(cos_theta, xi, xi_next, d, V):

    K, P, N = xi.shape

    for i in range(N):
        V[i] = 0.0

    for a in range(K):                  # sum_{a=1}^{K}   -- axis 0
        for mu in range(P):             # sum_{mu=1}^{P}  -- axis 1
            # Overlap of the current state with pattern (a, mu), computed once and
            # reused by all N neurons; the j != i correction is a cheap subtraction.
            overlap = 0.0
            for j in range(N):
                overlap += xi[a, mu, j] * cos_theta[j]
            for i in range(N):
                inner_sum = overlap - xi[a, mu, i] * cos_theta[i]   # drop the j == i term
                h = (inner_sum / (N - 1)) ** d
                V[i] += xi_next[a, mu, i] * h


@njit(cache=True, fastmath=True)
def _simulate_euler_multi(theta_init, omega, xi, xi_next, d, dt, num_steps,
                           stationary_eps=STATIONARY_EPS, stationary_patience=STATIONARY_PATIENCE):
    """Multi-sequence ODE kernel: `_simulate_euler`'s Euler loop and Stationary Break
    Check, driven by `_multi_drive`'s sum_a sum_mu coupling."""
    N = theta_init.shape[0]

    history = np.empty((num_steps + 1, N), dtype=np.float64)
    theta = theta_init.copy()
    history[0] = theta

    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    V = np.empty(N, dtype=np.float64)

    stationary_count = 0
    actual_steps = num_steps

    for step in range(num_steps):
        for i in range(N):
            cos_theta[i] = np.cos(theta[i])
            sin_theta[i] = np.sin(theta[i])

        _multi_drive(cos_theta, xi, xi_next, d, V)

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
def _simulate_maruyama_multi(theta_init, omega, xi, xi_next, d, dt, num_steps, phase_noise, seed,
                              stationary_eps=STATIONARY_EPS, stationary_patience=STATIONARY_PATIENCE):
    """Multi-sequence SDE kernel: same `_multi_drive` coupling as
    `_simulate_euler_multi`, with the per-step noise kick `_simulate_maruyama` uses."""
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

        _multi_drive(cos_theta, xi, xi_next, d, V)

        max_abs_dtheta = 0.0
        for i in range(N):
            dtheta_i = omega[i] - sin_theta[i] * V[i]
            random_shock = np.random.randn()
            theta[i] = theta[i] + (dt * dtheta_i) + (noise_scale * random_shock)
            if abs(dtheta_i) > max_abs_dtheta:
                max_abs_dtheta = abs(dtheta_i)

        history[step + 1] = theta.copy()

        # Same stationarity signal as the ODE case: the deterministic drift alone
        # (dtheta_i, ignoring the noise kick) having settled means further integration
        # is just wandering near a fixed point.
        if max_abs_dtheta < stationary_eps:
            stationary_count += 1
            if stationary_count >= stationary_patience:
                actual_steps = step + 1
                break
        else:
            stationary_count = 0

    return history[:actual_steps + 1]


# ---------------------------------------------------------------------------
# Network: config + omega + simulation + diagnostics + plotting
# ---------------------------------------------------------------------------

class KuramotoNetwork:
    """High-order Kuramoto dense-associative-memory network.

    Owns simulation parameters, intrinsic frequencies, and ODE/SDE integration +
    overlap diagnostics + plotting. It does not own any patterns itself -- every method
    below takes a `Sequence` or `MultiSequence` explicitly, the same way the kernels
    above take `xi`/`xi_next` explicitly rather than reading them off `self`. There is
    no separate config object either -- the run parameters are just attributes of the
    network, set from whatever the caller (typically a notebook) passes in.

    `frequency_std` (heterogeneity of the intrinsic frequencies `omega_i`) defaults to
    0.0, i.e. no natural-frequency drift -- the literal `dtheta_i/dt = -sin(theta_i) *
    sum_mu ...` update. Set it > 0 to add omega_i ~ Normal(`frequency_mean`,
    `frequency_std`) as a persistent per-oscillator drift term, for exploring robustness
    to disorder. `frequency_mean` (default 0.0) shifts that draw.
    """

    def __init__(
        self,
        N: int,
        seed: int,
        dt: float,
        T: float,
        phase_noise: float,
        d: int,
        corruption_rate: float,
        tolerance: float,
        mode: str,
        frequency_std: float = 0.0,
        frequency_mean: float = 0.0,
    ):
        self.N = N
        self.seed = seed
        self.dt = dt
        self.T = T
        self.phase_noise = phase_noise
        self.d = d
        self.corruption_rate = corruption_rate
        self.tolerance = tolerance
        self.mode = mode
        self.frequency_std = frequency_std
        self.frequency_mean = frequency_mean

        self.omega = self._make_omega()

    # -- setup -----------------------------------------------------------------

    def num_steps_for(self, T: float) -> int:
        return int(T / self.dt)

    def _make_omega(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.normal(self.frequency_mean, self.frequency_std, self.N)

    # -- generation convenience (defaults only; the logic lives in the free
    #    functions above) -------------------------------------------------------

    def generate_sequence(
        self,
        seq_len: int,
        corruption_rate: Optional[float] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Sequence:
        return generate_sequence(
            self.N, seq_len, _default(corruption_rate, self.corruption_rate),
            name=name, seed=_default(seed, self.seed),
        )

    def generate_multi_sequence(
        self,
        seq_len: int,
        names: Optional[TypingSequence[Optional[str]]] = None,
        count: Optional[int] = None,
        corruption_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> MultiSequence:
        return generate_multi_sequence(
            self.N, seq_len, _default(corruption_rate, self.corruption_rate),
            names=names, count=count, seed=_default(seed, self.seed),
        )

    # -- simulation --------------------------------------------------------

    def _integrate(
        self,
        sequences,
        theta_init: np.ndarray,
        mode: Optional[str],
        seed: Optional[int],
        T: Optional[float],
        boundary: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Shared ODE/SDE dispatch: resolve per-call overrides against the network's
        config defaults, run the requested kernel, and return (theta_history, time).

        Takes the `Sequence`/`MultiSequence` itself rather than pre-extracted arrays and
        pulls the (patterns, successors) pair off it here -- both are fully determined by
        `sequences` and `boundary`, so there's no reason for the caller to unpack them
        only to hand both halves back.

        A `MultiSequence` yields the rectangular (K, P, N) tensors and runs the
        multi-sequence kernels; a lone `Sequence` yields its (P, N) arrays and runs the
        single-sequence ones. That choice is made once, up front, rather than re-derived
        from `xi.ndim` inside each `mode` branch.

        The kernels may return fewer than num_steps+1 rows if the Stationary Break
        Check ended the integration early (see STATIONARY_EPS/STATIONARY_PATIENCE), so
        `time` is built from however many rows actually came back (`actual_steps * dt`)
        rather than assumed to span the originally-requested `T`."""
        mode = _default(mode, self.mode)
        seed = _default(seed, self.seed)
        T = _default(T, self.T)
        num_steps = self.num_steps_for(T)

        # One question asked once: a MultiSequence supplies the (K, P, N) tensors and the
        # kernels that sum over (a, mu); a lone Sequence supplies (P, N) and the kernels
        # that sum over mu. `mode` then picks which of that pair to call.
        if isinstance(sequences, MultiSequence):
            xi, xi_next = sequences.xi_tensor(), sequences.xi_next_tensor(boundary=boundary)
            ode_kernel, sde_kernel = _simulate_euler_multi, _simulate_maruyama_multi
        else:
            xi, xi_next = sequences.xi(), sequences.xi_next(boundary=boundary)
            ode_kernel, sde_kernel = _simulate_euler, _simulate_maruyama

        if mode == "ode":
            theta_history = ode_kernel(
                theta_init, self.omega, xi, xi_next, self.d, self.dt, num_steps,
            )
        elif mode == "sde":
            theta_history = sde_kernel(
                theta_init, self.omega, xi, xi_next, self.d, self.dt, num_steps,
                self.phase_noise, seed,
            )
        else:
            raise ValueError("mode must be either 'ode' or 'sde'")

        actual_steps = theta_history.shape[0] - 1
        time = np.linspace(0.0, actual_steps * self.dt, actual_steps + 1)
        return theta_history, time

    @staticmethod
    def _default_cue_source(sequences) -> Sequence:
        """Default cue source when the caller doesn't pass one explicitly: the
        sequence itself if `sequences` is a lone `Sequence`, or the first member added
        if it's a `MultiSequence`."""
        return sequences if isinstance(sequences, Sequence) else sequences[0]

    def simulate(
        self,
        sequences,
        cue: Optional[Sequence] = None,
        cue_idx: int = 0,
        mode: Optional[str] = None,
        seed: Optional[int] = None,
        T: Optional[float] = None,
        boundary: Optional[str] = None,
    ) -> dict:
        """Run one trial, driven by the successor-pattern tensor of `sequences` (a
        `Sequence` for single-sequence recall, or a `MultiSequence` for several at
        once -- cross-sequence interference included, since they're driven together).

        Parameters
        ----------
        cue : the `Sequence` object to cue from -- since a `Sequence` already carries
              its own name, this is the object itself, not a name to look up. Defaults
              to `sequences` (or its first member, for a `MultiSequence`); pass a
              different `Sequence` explicitly for cross-sequence cueing experiments.
              `cue_idx` (default 0, i.e. that sequence's first pattern) selects which
              of its own patterns is used as the initial condition theta(0), taken
              uncorrupted -- the cue is never corrupted here, only the sequence
              *generation* step (mutation) is.
        seed : override config defaults for this trial.
        T : override the network's configured `self.T` for this trial only. Useful when
            sweeping sequence length, e.g. `net.simulate(..., T=dwell_time_per_transition * seq_len)`.
        boundary : what each sequence's last pattern transitions to -- None (default) or
                   "self" (xi^{P+1} == xi^P: settles at the last pattern) or "cycle"
                   (xi^{P+1} == xi^1: wraps back to the first, per-sequence). See
                   `Sequence.xi_next`.

        A `MultiSequence` is driven by the multi-sequence kernels, over the rectangular
        (K, P, N) tensors from `xi_tensor()`/`xi_next_tensor()`; every stored sequence
        must therefore have the same length P (`MultiSequence` raises otherwise). A lone
        `Sequence` takes the single-sequence kernels over its own (P, N) arrays, exactly
        as before.

        Returns
        -------
        dict with 'time', 'theta_history', 'cue' (= (cue.name, cue_idx)), and
        'overlaps' (every pattern in `sequences`). For a `MultiSequence`, also
        'sequence_overlaps' ({name: overlap_array} restricted to each member's own
        patterns).
        """
        cue_seq = _default(cue, self._default_cue_source(sequences))

        theta_init = cue_seq.cue(cue_idx, corruption_rate=0.0)
        theta_history, time = self._integrate(sequences, theta_init, mode, seed, T, boundary)

        result = {"time": time, "theta_history": theta_history, "cue": (cue_seq.name, cue_idx)}
        result["overlaps"] = self.overlaps(theta_history, sequences)
        if isinstance(sequences, MultiSequence):
            result["sequence_overlaps"] = {name: self.overlaps(theta_history, sequences[name]) for name in sequences.names}
        return result

    # -- diagnostics ---------------------------------------------------------

    def overlaps(self, theta_history: np.ndarray, sequences) -> np.ndarray:
        """Global overlap m^mu(t) = (1/N) sum_i xi_i^mu cos(theta_i(t)) for every
        pattern in `sequences`. Shape: (num_steps+1, num_patterns_in_sequences)."""
        xi = sequences.xi()
        return np.cos(theta_history) @ xi.T / xi.shape[1]

    # -- sequence decoding ---------------------------------------------------

    def decode_sequence_peaks(
        self,
        result: dict,
        sequences,
        min_distance,
        tolerance: Optional[float] = None,
    ) -> List[Tuple[str, int]]:
        """Offline/batch decoder: find each pattern's tallest hump (`scipy.signal.find_peaks`)
        and order patterns by when their peak occurs.

        A pattern only counts as "recovered" if its peak overlap clears `tolerance`
        (default: the network's configured `self.tolerance`); patterns with no
        qualifying peak are dropped from the output. If a pattern has several peaks
        above tolerance, only its highest one is used, so each pattern contributes at
        most one slot to the retrieved sequence.

        The cue pattern (peaking at t=0) and whichever pattern the trajectory ends up
        at by t=T (with the default "self" boundary, that's the terminal pattern,
        since it settles there) are boundary samples with no point before/after them
        to fall away from, so plain `find_peaks` would never flag them as peaks. Each
        series is padded with -inf on both ends before peak detection so a boundary
        sample still counts as a peak if it dominates its one interior neighbor.
        """
        tolerance = _default(tolerance, self.tolerance)
        labels = sequences.labels()
        overlap_history = self.overlaps(result["theta_history"], sequences)

        best_peak: Dict[Tuple[str, int], Tuple[int, float]] = {}  # label -> (time_idx, height)
        for mu, label in enumerate(labels):
            padded = np.concatenate(([-np.inf], overlap_history[:, mu], [-np.inf]))
            peak_idx, props = find_peaks(padded, height=tolerance, distance=min_distance)
            if peak_idx.size == 0:
                continue
            best = int(np.argmax(props["peak_heights"]))
            best_peak[label] = (int(peak_idx[best]) - 1, float(props["peak_heights"][best]))

        ordered = sorted(best_peak.items(), key=lambda kv: kv[1][0])
        return [label for label, _ in ordered]

    def decode_sequence_online(
        self,
        result: dict,
        sequences,
        margin: float,
        tolerance: Optional[float] = None
    ) -> List[Tuple[str, int]]:
        """Online/streaming decoder: a sticky winner-take-all state machine.

        At each timestep, the current label only switches to a challenger pattern once
        the challenger's overlap clears `tolerance` (default: the network's configured
        `self.tolerance`) AND beats the current label's overlap by at least `margin`
        (hysteresis -- avoids flicker from noisy near-ties between crossing curves).
        Unlike `decode_sequence_peaks`, this only ever looks at overlaps up to the
        current timestep, so it also works on a partial/live trace.
        """
        tolerance = _default(tolerance, self.tolerance)
        labels = sequences.labels()
        overlap_history = self.overlaps(result["theta_history"], sequences)

        retrieved: List[Tuple[str, int]] = []
        current_idx: Optional[int] = None
        for t in range(overlap_history.shape[0]):
            row = overlap_history[t]
            candidate_idx = int(np.argmax(row))
            candidate_val = row[candidate_idx]

            if candidate_val < tolerance:
                continue

            if current_idx is None:
                current_idx = candidate_idx
                retrieved.append(labels[current_idx])
                continue

            if candidate_idx != current_idx and candidate_val - row[current_idx] > margin:
                current_idx = candidate_idx
                retrieved.append(labels[current_idx])

        return retrieved

    def compare_to_stored_sequence(self, retrieved: TypingSequence[Tuple[str, int]], sequences) -> int:
        """Compare a decoded/retrieved sequence against the stored ground-truth order
        (`sequences.labels()`), position by position.

        Returns the 0-indexed step at which recall first fails (`retrieved[step] !=
        expected[step]`, or `retrieved` ran out early) -- i.e. how many leading steps
        were recalled correctly before the sequence broke. Returns `len(expected)` if
        `retrieved` matches exactly (perfect recall).
        """
        expected = sequences.labels()
        for step, label in enumerate(expected):
            if step >= len(retrieved) or retrieved[step] != label:
                return step
        return len(expected)

    # -- plotting --------------------------------------------------------------

    def plot(self, result: dict, sequences, title: Optional[str] = None, ax=None):
        """Plot overlap with each pattern in `sequences` over time.

        Pass an existing `ax` (e.g. one panel of `plt.subplots(...)`) to draw into a shared,
        compact multi-panel figure instead of popping a full-size figure per call."""
        import matplotlib.pyplot as plt

        labels = sequences.labels()
        overlap_history = self.overlaps(result["theta_history"], sequences)

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(10, 5))

        for mu, (name, idx) in enumerate(labels):
            ax.plot(result["time"], overlap_history[:, mu], linewidth=2.0, label=f"{name}[{idx}]")
        cue_name, cue_idx = result["cue"]
        ax.set_title(title or f"Memory pattern overlaps (cued from corrupted '{cue_name}[{cue_idx}]')")
        ax.set_xlabel("Time (t)")
        ax.set_ylabel(r"Overlap $m^\mu(t)$")
        ax.legend(fontsize="small", ncol=3)
        ax.grid(True)

        if standalone:
            plt.show()


# ---------------------------------------------------------------------------
# Example usage (only runs when executed directly, not on import)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # All run parameters come from one explicit config -- there are no network-side
    # defaults for these, precisely so a notebook's CONFIG dict is the single source
    # of truth and can't silently diverge from what the network assumes. N and seed
    # are independent variables (swept across trials), so they're passed separately.
    CONFIG = dict(dt=0.02, T=55.0, frequency_mean=0.0, frequency_std=0.03, phase_noise=0.01,
                  d=3, corruption_rate=0.2, tolerance=0.4, mode="ode")

    net = KuramotoNetwork(N=40, seed=10, **CONFIG)
    A = net.generate_sequence(name="A", seq_len=5)
    result = net.simulate(A)
    net.plot(result, A)
