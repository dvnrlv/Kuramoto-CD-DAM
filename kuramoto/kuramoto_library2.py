"""
High-Order Kuramoto Associative Memory Network (OOP)
=====================================================

Object-oriented rewrite of `high_order_kuramoto.ipynb`.

Model
-----
Single sequence (period P, terminal self-loop xi^{P+1} == xi^P -- the last pattern points
to itself rather than wrapping back to xi^1, so the dynamics settle at the last pattern
instead of cycling indefinitely):

    dtheta_i/dt = omega_i - sin(theta_i) * sum_mu xi_i^{mu+1} * ( (1/(N-1)) sum_{j!=i} xi_j^mu cos(theta_j) )^d

The sum over mu is implemented as a flat list of (from -> to) transition edges: P-1
consecutive-pair edges plus one self-loop edge at the end, fed into the same simulation
core here (`_simulate_euler` / `_simulate_maruyama`).

Two classes:

- `PatternLibrary`   : owns pattern generation (xi in {-1,+1}^N), naming, the single
                       stored sequence (which pattern indices, in order), edge-building,
                       and corruption of a cue.
- `KuramotoNetwork`  : owns the run config (dt, T, d, noise, mode, ...) as plain attributes
                       (no separate config class -- the caller, typically a notebook,
                       supplies these as keyword args or an unpacked dict), intrinsic
                       frequencies omega, and runs ODE/SDE simulations + overlap
                       diagnostics + plotting against a PatternLibrary.

`N` (network size) and `seed` (RNG seed) are deliberately kept out of the shared config:
both are independent variables you sweep across trials/experiments, not fixed run
parameters, so they're passed to `KuramotoNetwork(...)` explicitly alongside the config
rather than living inside it.

Typical usage
-------------
    CONFIG = dict(dt=0.02, T=55.0, frequency_std=0.03, phase_noise=0.01,
                  d=3, corruption_rate=0.2, mode="ode")
    net = KuramotoNetwork(N=40, seed=10, **CONFIG)
    seq_len = 5  # sequences can be any length, so names are generated, never hand-typed
    names = [f"A_{k}" for k in range(1, seq_len + 1)]
    net.generate_patterns(names)
    net.add_sequence(names)
    result = net.simulate(cue_pattern=names[0])
    net.plot(result)

Sweeping N or seed across trials just means constructing a new `KuramotoNetwork` per
value while reusing the same `CONFIG` -- that sweep loop is left for you to write in a
notebook, not baked into this module.

Implementation note
--------------------
Public method names and signatures are unchanged from the original module (external
code depends on them). Internally, the repeated `x = fallback if x is None else x`
idiom has been collected into a single `_default()` helper, and the logic that used to
be duplicated between the single-sequence and multi-sequence code paths (mutation
chains, edge-building, ODE/SDE dispatch) has been factored into small private helpers
that both paths call.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scipy.signal import find_peaks

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - allows the module to still import without numba
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrap(fn):
            return fn
        return wrap


def _default(value, fallback):
    """Return `value` unless it's None, in which case return `fallback`.

    Collects the `x = fallback if x is None else x` idiom (used throughout this module
    for "per-call override of a config default") into one place.
    """
    return fallback if value is None else value


# ---------------------------------------------------------------------------
# Low-level numba kernels (free functions -- numba methods can't take `self`)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _simulate_euler(theta_init, omega, xi, edges_from, edges_to, d, dt, num_steps):
    N = theta_init.shape[0]
    num_patterns = xi.shape[0]
    num_edges = edges_from.shape[0]

    history = np.empty((num_steps + 1, N), dtype=np.float64)
    theta = theta_init.copy()
    history[0] = theta

    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    S = np.empty(num_patterns, dtype=np.float64)

    for step in range(num_steps):
        for i in range(N):
            cos_theta[i] = np.cos(theta[i])
            sin_theta[i] = np.sin(theta[i])

        # Overlap with every stored pattern, computed once and reused across edges
        for p in range(num_patterns):
            temp_sum = 0.0
            for j in range(N):
                temp_sum += xi[p, j] * cos_theta[j]
            S[p] = temp_sum

        V = np.zeros(N, dtype=np.float64)
        for e in range(num_edges):
            frm = edges_from[e]
            to = edges_to[e]
            for i in range(N):
                inner_sum = S[frm] - xi[frm, i] * cos_theta[i]
                h = (inner_sum / (N - 1)) ** d
                V[i] += xi[to, i] * h

        for i in range(N):
            dtheta_i = omega[i] - sin_theta[i] * V[i]
            theta[i] = theta[i] + dt * dtheta_i

        history[step + 1] = theta.copy()

    return history


@njit(cache=True)
def _simulate_maruyama(theta_init, omega, xi, edges_from, edges_to, d, dt, num_steps, phase_noise, seed):
    np.random.seed(seed) # for @njit, we have to use np.random.seed inside

    N = theta_init.shape[0]
    num_patterns = xi.shape[0]
    num_edges = edges_from.shape[0]

    history = np.empty((num_steps + 1, N), dtype=np.float64)
    theta = theta_init.copy()
    history[0] = theta

    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    S = np.empty(num_patterns, dtype=np.float64)

    noise_scale = phase_noise * np.sqrt(dt)

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
        for e in range(num_edges):
            frm = edges_from[e]
            to = edges_to[e]
            for i in range(N):
                inner_sum = S[frm] - xi[frm, i] * cos_theta[i]
                h = (inner_sum / (N - 1)) ** d
                V[i] += xi[to, i] * h

        for i in range(N):
            dtheta_i = omega[i] - sin_theta[i] * V[i]
            random_shock = np.random.randn()
            theta[i] = theta[i] + (dt * dtheta_i) + (noise_scale * random_shock)

        history[step + 1] = theta.copy()

    return history


# ---------------------------------------------------------------------------
# Pattern / sequence bookkeeping
# ---------------------------------------------------------------------------

class PatternLibrary:
    """Owns the stored binary phase patterns (xi in {-1,+1}^N) and the single stored sequence.

    A "sequence" is an ordered list of pattern indices (the mu -> mu+1 chain the dynamics
    follow, with a terminal self-loop at the end -- see module docstring).
    """

    def __init__(self, N: int, seed: int = 10):
        self.N = N
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.pattern_names: List[str] = []
        self.name_to_index: Dict[str, int] = {}
        self.phases: np.ndarray = np.empty((0, N))  # (num_patterns, N), angles in {0, pi}
        self.xi: np.ndarray = np.empty((0, N))       # (num_patterns, N), spins in {-1, +1}

        self.sequence: Optional[np.ndarray] = None  # array of pattern indices, in order

        # Named sequences (multi-sequence recall, section 2): sequence name -> array of
        # pattern indices, in order. `self.sequence` above is the single-sequence case and
        # is left untouched by any of this -- `self.sequences` is a separate, parallel
        # store, not a generalization that subsumes it. A pattern is just a row of `xi`
        # (and its name a key into `name_to_index`); which named sequence(s) it belongs to
        # is purely a matter of which index arrays here happen to reference that row, so
        # the same pattern (e.g. a shared first pattern across several sequences) can sit
        # inside more than one entry of `self.sequences` without being duplicated in `xi`.
        self.sequences: Dict[str, np.ndarray] = {}

    # -- pattern generation -------------------------------------------------

    def generate_patterns(self, names: Sequence[str]) -> None:
        """Append `len(names)` new random binary patterns under the given names."""
        new_names = list(names)
        overlap = set(new_names) & set(self.pattern_names)
        if overlap:
            raise ValueError(f"Pattern name(s) already exist: {sorted(overlap)}")

        new_phases = self._rng.choice([0.0, np.pi], size=(len(new_names), self.N))
        new_xi = np.cos(new_phases)

        self.phases = np.vstack([self.phases, new_phases]) if self.phases.size else new_phases
        self.xi = np.vstack([self.xi, new_xi]) if self.xi.size else new_xi

        start = len(self.pattern_names)
        for offset, name in enumerate(new_names):
            self.name_to_index[name] = start + offset
        self.pattern_names.extend(new_names)

    def index_of(self, name: str) -> int:
        return self.name_to_index[name]

    # -- mutation-based pattern generation --------------------------------------

    def _append_pattern(self, name: str, phase: np.ndarray) -> None:
        """Store one already-computed phase vector under `name`. The single place that
        knows how to grow `phases`/`xi`/`name_to_index`/`pattern_names` together for a
        one-at-a-time addition (`generate_patterns` does the analogous thing in bulk,
        for many patterns at once, so it keeps its own batched vstack instead of calling
        this in a loop)."""
        if name in self.name_to_index:
            raise ValueError(f"Pattern name already exists: {name}")
        self.name_to_index[name] = len(self.pattern_names)
        self.pattern_names.append(name)
        self.phases = np.vstack([self.phases, phase[None, :]]) if self.phases.size else phase[None, :]
        new_xi = np.cos(phase)
        self.xi = np.vstack([self.xi, new_xi[None, :]]) if self.xi.size else new_xi[None, :]

    def mutate_pattern(self, base_name: str, new_name: str, corruption_rate: float, seed: Optional[int] = None) -> np.ndarray:
        """Corrupt `base_name` (reuses `corrupt`, i.e. the same random-subset-of-spins
        flip 0 <-> pi used to build a cue), store the result under `new_name`, and return
        the new phase vector. Unlike `generate_patterns`, the new pattern is correlated
        with its base rather than independent random noise -- this is the building block
        for a mutation-driven sequence (xi^{k+1} = mutate(xi^k)).

        `corrupt` itself is pure (reads `self.phases`, doesn't write it); `_append_pattern`
        is the only step here that mutates `self` -- so this method reads as "compute, then
        store, then hand back what was stored" rather than hand-rolling the bookkeeping."""
        new_phase = self.corrupt(base_name, corruption_rate, seed=seed)
        self._append_pattern(new_name, new_phase)
        return new_phase

    def _mutation_chain(
        self,
        base_name: str,
        seq_len: int,
        corruption_rate: float,
        seed: Optional[int],
        name_template: str,
    ) -> List[str]:
        """Shared core of `generate_sequence`/`generate_named_sequence`: build a length-
        (seq_len+1) chain of pattern names by repeatedly mutating `base_name` (xi^1 =
        base_name, already generated; xi^{k+1} = mutate(xi^k, corruption_rate)). Only
        generates the names/patterns -- registering the chain as a sequence is left to
        the caller, since the two callers register it under different stores
        (`self.sequence` vs. `self.sequences[name]`)."""
        rng = np.random.default_rng(_default(seed, self.seed))
        names = [base_name]
        current = base_name
        for k in range(1, seq_len + 1):
            new_name = name_template.format(base=base_name, k=k)
            step_seed = int(rng.integers(0, 2**32 - 1))
            self.mutate_pattern(current, new_name, corruption_rate, seed=step_seed)
            names.append(new_name)
            current = new_name
        return names

    def generate_sequence(
        self,
        base_name: str,
        seq_len: int,
        corruption_rate: float,
        seed: Optional[int] = None,
        name_template: str = "{base}_{k}",
    ) -> List[str]:
        """Build a length-(num_mutations+1) sequence by repeatedly mutating `base_name`:
        xi^1 = base_name (already generated), xi^{k+1} = mutate(xi^k, corruption_rate).
        Registers the resulting chain as the stored sequence and returns the ordered
        pattern names."""
        names = self._mutation_chain(base_name, seq_len, corruption_rate, seed, name_template)
        self.add_sequence(names)
        return names

    # -- sequence bookkeeping -------------------------------------------------

    def add_sequence(self, pattern_names_in_order: Sequence[str]) -> None:
        """Register the (single) stored sequence over already-generated pattern names."""
        self.sequence = self._resolve_indices(pattern_names_in_order)

    def _resolve_indices(self, pattern_names_in_order: Sequence[str]) -> np.ndarray:
        """Look up an ordered list of pattern names as an int64 index array, raising a
        clear error for any name that hasn't been generated yet."""
        missing = [n for n in pattern_names_in_order if n not in self.name_to_index]
        if missing:
            raise ValueError(f"Unknown pattern name(s): {missing}. Call generate_patterns first.")
        return np.array([self.name_to_index[n] for n in pattern_names_in_order], dtype=np.int64)

    @staticmethod
    def _chain_edges(seq_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(from, to) transition edges for one ordered chain of pattern indices: one edge
        per consecutive pair, plus a terminal self-loop (last index -> itself) so the
        dynamics settle at the last pattern instead of cycling back to the first."""
        edges_from = np.concatenate([seq_idx[:-1], seq_idx[-1:]])
        edges_to = np.concatenate([seq_idx[1:], seq_idx[-1:]])
        return edges_from.astype(np.int64), edges_to.astype(np.int64)

    def build_transition_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Flatten the stored sequence's mu -> mu+1 transitions into a (from, to) edge
        list. The final transition self-loops (xi^{P+1} == xi^P) rather than wrapping back
        to the first pattern, so the dynamics settle at the last pattern instead of
        cycling indefinitely."""
        if self.sequence is None:
            raise ValueError("No sequence registered; call add_sequence(...) first.")
        return self._chain_edges(self.sequence)

    # -- multiple named sequences (section 2) --------------------------------

    def add_named_sequence(self, name: str, pattern_names_in_order: Sequence[str]) -> None:
        """Register one named sequence xi^{a,1..Pa} over already-generated pattern names
        (the multi-sequence analogue of `add_sequence`, keyed by `name` instead of being
        the single implicit `self.sequence`)."""
        if name in self.sequences:
            raise ValueError(f"Sequence name already exists: {name}")
        self.sequences[name] = self._resolve_indices(pattern_names_in_order)

    def generate_named_sequence(
        self,
        seq_name: str,
        base_name: str,
        seq_len: int,
        corruption_rate: float,
        seed: Optional[int] = None,
        name_template: str = "{base}_{k}",
    ) -> List[str]:
        """Same mutation mechanics as `generate_sequence` (xi^{a,1} = base_name, already
        generated; xi^{a,k+1} = mutate(xi^{a,k}, corruption_rate)), but registers the
        resulting chain under `seq_name` in `self.sequences` instead of overwriting the
        single `self.sequence`."""
        names = self._mutation_chain(base_name, seq_len, corruption_rate, seed, name_template)
        self.add_named_sequence(seq_name, names)
        return names

    def generate_sequences(
        self,
        base_names: Sequence[str],
        seq_len: int,
        corruption_rate: float,
        seed: Optional[int] = None,
        name_template: str = "{base}_{k}",
    ) -> Dict[str, List[str]]:
        """Generate M independent sequences at once: one fresh random base pattern per
        name in `base_names` (unlike a single mutation chain, the M base patterns here are
        drawn independently of each other, not derived from one shared ancestor), each
        then mutated `seq_len` times into its own named chain under `base_names[a]`.
        Returns `{sequence_name: ordered_pattern_names}`."""
        rng = np.random.default_rng(_default(seed, self.seed))
        base_names = list(base_names)
        self.generate_patterns(base_names)

        result: Dict[str, List[str]] = {}
        for base_name in base_names:
            seq_seed = int(rng.integers(0, 2**32 - 1))
            result[base_name] = self.generate_named_sequence(
                seq_name=base_name, base_name=base_name, seq_len=seq_len,
                corruption_rate=corruption_rate, seed=seq_seed, name_template=name_template,
            )
        return result

    def pattern_names_for_sequence(self, name: str) -> List[str]:
        """Ordered pattern names belonging to named sequence `name` -- the multi-sequence
        counterpart of reading `self.sequence` directly, restricted to one sequence."""
        if name not in self.sequences:
            raise ValueError(f"Unknown sequence name: {name!r}")
        return [self.pattern_names[i] for i in self.sequences[name]]

    def build_transition_edges_for(
        self, sequence_names: Optional[Sequence[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Flatten every named sequence's mu -> mu+1 transitions into one combined (from,
        to) edge list (default: every stored sequence). Each sequence keeps its own
        terminal self-loop, matching `build_transition_edges`'s single-sequence
        convention. A pattern shared by multiple sequences (e.g. a common first pattern)
        contributes one edge per sequence it starts a transition in, so summing the
        resulting edge list reproduces `sum_a sum_mu (...)` from the update rule exactly."""
        names = _default(list(sequence_names) if sequence_names is not None else None,
                          list(self.sequences.keys()))
        if not names:
            raise ValueError("No sequences registered; call add_named_sequence(...) first.")

        missing = [n for n in names if n not in self.sequences]
        if missing:
            raise ValueError(f"Unknown sequence name(s): {missing}")

        edge_pairs = [self._chain_edges(self.sequences[n]) for n in names]
        edges_from = np.concatenate([frm for frm, _ in edge_pairs])
        edges_to = np.concatenate([to for _, to in edge_pairs])
        return edges_from, edges_to

    # -- cueing ----------------------------------------------------------------

    def corrupt(self, pattern_name: str, corruption_rate: float, seed: Optional[int] = None) -> np.ndarray:
        """Return a corrupted copy (random subset of neurons flipped 0 <-> pi) of a
        pattern's phase vector, for use as an initial condition theta(0)."""
        idx = self.index_of(pattern_name)
        phase_pattern = self.phases[idx].copy()
        if corruption_rate <= 0:
            return phase_pattern

        rng = np.random.default_rng(_default(seed, self.seed))
        num_flips = int(corruption_rate * phase_pattern.shape[0])
        flip_idx = rng.choice(phase_pattern.shape[0], size=num_flips, replace=False)
        phase_pattern[flip_idx] = np.pi - phase_pattern[flip_idx]
        return phase_pattern


# ---------------------------------------------------------------------------
# Network: config + omega + simulation + diagnostics + plotting
# ---------------------------------------------------------------------------

class KuramotoNetwork:
    """High-order Kuramoto dense-associative-memory network.

    Wraps a `PatternLibrary` with simulation parameters, intrinsic frequencies, and
    ODE/SDE integration + overlap diagnostics + plotting. There is no separate config
    object -- the run parameters below are just attributes of the network, set from
    whatever the caller (typically a notebook) passes in.
    """

    def __init__(
        self,
        N: int,
        seed: int,
        dt: float,
        T: float,
        frequency_std: float,
        phase_noise: float,
        d: int,
        corruption_rate: float,
        tolerance: float,
        mode: str,
        plibrary: Optional[PatternLibrary] = None,
    ):
        self.N = N
        self.seed = seed
        self.dt = dt
        self.T = T
        self.frequency_std = frequency_std
        self.phase_noise = phase_noise
        self.d = d
        self.corruption_rate = corruption_rate
        self.tolerance = tolerance
        self.mode = mode

        self.plibrary = plibrary or PatternLibrary(N=self.N, seed=self.seed)
        if self.plibrary.N != self.N:
            raise ValueError("PatternLibrary.N must match KuramotoNetwork.N") # raise errors if mismatched number of neurons 
        self.omega = self._make_omega()

    # -- setup -----------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return self.num_steps_for(self.T)

    def num_steps_for(self, T: float) -> int:
        return int(T / self.dt)

    def time_vector(self, T: Optional[float] = None) -> np.ndarray:
        T = _default(T, self.T)
        return np.linspace(0.0, T, self.num_steps_for(T) + 1)

    def _make_omega(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, self.frequency_std, self.N)

    def generate_patterns(self, names: Sequence[str]) -> None:
        self.plibrary.generate_patterns(names)

    def add_sequence(self, pattern_names_in_order: Sequence[str]) -> None:
        self.plibrary.add_sequence(pattern_names_in_order)

    def add_named_sequence(self, name: str, pattern_names_in_order: Sequence[str]) -> None:
        self.plibrary.add_named_sequence(name, pattern_names_in_order)

    def generate_sequences(
        self,
        base_names: Sequence[str],
        seq_len: int,
        corruption_rate: Optional[float] = None,
        seed: Optional[int] = None,
        name_template: str = "{base}_{k}",
    ) -> Dict[str, List[str]]:
        return self.plibrary.generate_sequences(
            base_names, seq_len,
            _default(corruption_rate, self.corruption_rate),
            seed=_default(seed, self.seed),
            name_template=name_template,
        )

    # -- simulation --------------------------------------------------------

    def _integrate(
        self,
        edges_from: np.ndarray,
        edges_to: np.ndarray,
        theta_init: np.ndarray,
        mode: Optional[str],
        seed: Optional[int],
        T: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Shared ODE/SDE dispatch used by both `simulate` and `simulate_multi`: resolve
        the per-call overrides against the network's config defaults, run the requested
        kernel, and return (theta_history, time_vector). The kernels themselves don't
        care whether `edges_from`/`edges_to` came from one chain or several concatenated
        chains -- see `PatternLibrary.build_transition_edges[_for]`."""
        mode = _default(mode, self.mode)
        seed = _default(seed, self.seed)
        T = _default(T, self.T)
        num_steps = self.num_steps_for(T)

        if mode == "ode":
            theta_history = _simulate_euler(
                theta_init, self.omega, self.plibrary.xi, edges_from, edges_to,
                self.d, self.dt, num_steps,
            )
        elif mode == "sde":
            theta_history = _simulate_maruyama(
                theta_init, self.omega, self.plibrary.xi, edges_from, edges_to,
                self.d, self.dt, num_steps, self.phase_noise, seed,
            )
        else:
            raise ValueError("mode must be either 'ode' or 'sde'")

        return theta_history, self.time_vector(T)

    def simulate(
        self,
        mode: Optional[str] = None,
        cue_pattern: Optional[str] = None,
        seed: Optional[int] = None,
        T: Optional[float] = None,
    ) -> dict:
        """Run one trial.

        Parameters
        ----------
        mode : 'ode' (deterministic Forward Euler) or 'sde' (Euler-Maruyama with phase noise).
               Defaults to the network's configured `self.mode`.
        cue_pattern : name of the pattern to use as the initial condition theta(0), taken
                      uncorrupted -- the cue is never corrupted here, only the sequence
                      *generation* step (`generate_sequence`'s mutation) is. Defaults to the
                      first pattern of the stored sequence, so the default experiment is
                      "start exactly at the sequence's first pattern and see whether the
                      dynamics alone walk the chain forward."
        seed : override config defaults for this trial.
        T : override the network's configured `self.T` for this trial only (`self.T` is left
            untouched). Useful when sweeping sequence length: the per-transition time budget
            shrinks as the chain gets longer unless `T` is scaled up to match, e.g.
            `net.simulate(..., T=dwell_time_per_transition * seq_len)`.

        Returns
        -------
        dict with 'time', 'theta_history', 'cue', plus (for convenience) 'overlaps'
        (overlap with every stored pattern over time).
        """
        if self.plibrary.sequence is None:
            raise ValueError("No sequence registered; call add_sequence(...) first.")
        cue_pattern = _default(cue_pattern, self.plibrary.pattern_names[self.plibrary.sequence[0]])

        edges_from, edges_to = self.plibrary.build_transition_edges()
        theta_init = self.plibrary.corrupt(cue_pattern, corruption_rate=0.0)
        theta_history, time = self._integrate(edges_from, edges_to, theta_init, mode, seed, T)

        result = {"time": time, "theta_history": theta_history, "cue": cue_pattern}
        result["overlaps"] = self.overlaps(theta_history)
        return result

    def simulate_multi(
        self,
        cue_pattern: str,
        sequence_names: Optional[Sequence[str]] = None,
        mode: Optional[str] = None,
        seed: Optional[int] = None,
        T: Optional[float] = None,
    ) -> dict:
        """Run one multi-sequence trial: take `cue_pattern` uncorrupted as the initial
        condition theta(0), evolve the combined dynamics driven by every transition edge
        across `sequence_names` (default: every stored named sequence), and report overlap
        both against the whole library and per-sequence.

        `_simulate_euler`/`_simulate_maruyama` don't change at all for this -- they already
        just sum over a flat edge list, so `sum_a sum_mu (...)` in the multi-sequence update
        rule is exactly `sum_e (...)` over the edges `build_transition_edges_for` returns,
        the same mechanism `simulate()` already uses for a single chain.

        Unlike `simulate()`, `cue_pattern` has no sensible default here: with several
        sequences there's no one "first pattern" to fall back to, so it's required. That
        also makes cueing from a prefix pattern shared by multiple sequences an explicit
        choice (useful for studying cross-sequence interference, as in
        attempt0/high_order_kuramoto.ipynb section 2), not an accident. The cue is never
        corrupted here, only the sequence *generation* step (mutation) is.

        Returns
        -------
        dict with 'time', 'theta_history', 'cue', 'overlaps' (every stored pattern in the
        library, as in `simulate()`), and 'sequence_overlaps' -- {name: overlap_array} for
        each sequence in `sequence_names`, i.e. m^{a,mu}(t) restricted to sequence a's own
        patterns.
        """
        names = _default(list(sequence_names) if sequence_names is not None else None,
                          list(self.plibrary.sequences.keys()))

        edges_from, edges_to = self.plibrary.build_transition_edges_for(names)
        theta_init = self.plibrary.corrupt(cue_pattern, corruption_rate=0.0)
        theta_history, time = self._integrate(edges_from, edges_to, theta_init, mode, seed, T)

        result = {"time": time, "theta_history": theta_history, "cue": cue_pattern}
        result["overlaps"] = self.overlaps(theta_history)
        result["sequence_overlaps"] = {
            name: self.overlaps(theta_history, self.plibrary.pattern_names_for_sequence(name))
            for name in names
        }
        return result

    # -- diagnostics ---------------------------------------------------------

    def overlaps(self, theta_history: np.ndarray, pattern_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Global overlap m^mu(t) = (1/N) sum_i xi_i^mu cos(theta_i(t)) for chosen patterns
        (default: all stored patterns). Shape: (num_steps+1, num_patterns)."""
        if pattern_names is None:
            xi = self.plibrary.xi
        else:
            xi = self.plibrary.xi[[self.plibrary.index_of(n) for n in pattern_names]]
        return np.cos(theta_history) @ xi.T / xi.shape[1]

    # -- sequence decoding ---------------------------------------------------

    def decode_sequence_peaks(
        self,
        result: dict,
        tolerance: float,
        pattern_names: Optional[Sequence[str]] = None,
        min_distance: int = 1,
    ) -> List[str]:
        """Offline/batch decoder: find each pattern's tallest hump (`scipy.signal.find_peaks`)
        and order patterns by when their peak occurs.

        A pattern only counts as "recovered" if its peak overlap clears `tolerance`;
        patterns with no qualifying peak are dropped from the output. If a pattern has
        several peaks above tolerance, only its highest one is used, so each pattern
        contributes at most one slot to the retrieved sequence.

        The cue pattern (peaking at t=0) and the terminal pattern (settled at t=T, per
        the sequence's self-loop -- see module docstring) are boundary samples with no
        point before/after them to fall away from, so plain `find_peaks` would never
        flag them as peaks. Each series is padded with -inf on both ends before peak
        detection so a boundary sample still counts as a peak if it dominates its one
        interior neighbor.
        """
        names = _default(list(pattern_names) if pattern_names is not None else None,
                          self.plibrary.pattern_names)
        overlap_history = self.overlaps(result["theta_history"], names)

        best_peak: Dict[str, Tuple[int, float]] = {}  # name -> (time_idx, height)
        for mu, name in enumerate(names):
            padded = np.concatenate(([-np.inf], overlap_history[:, mu], [-np.inf]))
            peak_idx, props = find_peaks(padded, height=tolerance, distance=min_distance)
            if peak_idx.size == 0:
                continue
            best = int(np.argmax(props["peak_heights"]))
            best_peak[name] = (int(peak_idx[best]) - 1, float(props["peak_heights"][best]))

        ordered = sorted(best_peak.items(), key=lambda kv: kv[1][0])
        return [name for name, _ in ordered]

    def decode_sequence_online(
        self,
        result: dict,
        tolerance: float,
        margin: float = 0.0,
        pattern_names: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Online/streaming decoder: a sticky winner-take-all state machine.

        At each timestep, the current label only switches to a challenger pattern once
        the challenger's overlap clears `tolerance` AND beats the current label's overlap
        by at least `margin` (hysteresis -- avoids flicker from noisy near-ties between
        crossing curves). Unlike `decode_sequence_peaks`, this only ever looks at overlaps
        up to the current timestep, so it also works on a partial/live trace.
        """
        names = _default(list(pattern_names) if pattern_names is not None else None,
                          self.plibrary.pattern_names)
        overlap_history = self.overlaps(result["theta_history"], names)

        retrieved: List[str] = []
        current_idx: Optional[int] = None
        for t in range(overlap_history.shape[0]):
            row = overlap_history[t]
            candidate_idx = int(np.argmax(row))
            candidate_val = row[candidate_idx]

            if candidate_val < tolerance:
                continue

            if current_idx is None:
                current_idx = candidate_idx
                retrieved.append(names[current_idx])
                continue

            if candidate_idx != current_idx and candidate_val - row[current_idx] > margin:
                current_idx = candidate_idx
                retrieved.append(names[current_idx])

        return retrieved

    def compare_to_stored_sequence(self, retrieved: Sequence[str], sequence_name: Optional[str] = None) -> int:
        """Compare a decoded/retrieved sequence against the stored ground-truth sequence,
        position by position.

        sequence_name : which stored sequence to compare against. Defaults to the single
                         `self.plibrary.sequence` (unchanged single-sequence behaviour); pass
                         a name to compare against one of `self.plibrary.sequences` instead
                         (multi-sequence recall).

        Returns the 0-indexed step at which recall first fails (`retrieved[step] !=
        stored[step]`, or `retrieved` ran out early) -- i.e. how many leading steps were
        recalled correctly before the sequence broke. Returns `len(stored)` if `retrieved`
        matches the stored sequence exactly (perfect recall).
        """
        if sequence_name is None:
            if self.plibrary.sequence is None:
                raise ValueError("No sequence registered; call add_sequence(...) first.")
            stored = [self.plibrary.pattern_names[i] for i in self.plibrary.sequence]
        else:
            stored = self.plibrary.pattern_names_for_sequence(sequence_name)

        for step, expected in enumerate(stored):
            if step >= len(retrieved) or retrieved[step] != expected:
                return step
        return len(stored)

    # -- plotting --------------------------------------------------------------

    def plot(self, result: dict, pattern_names: Optional[Sequence[str]] = None, title: Optional[str] = None, ax=None):
        """Plot overlap with each stored (or chosen) pattern over time -- single-sequence style.

        Pass an existing `ax` (e.g. one panel of `plt.subplots(...)`) to draw into a shared,
        compact multi-panel figure instead of popping a full-size figure per call."""
        import matplotlib.pyplot as plt

        names = _default(pattern_names, self.plibrary.pattern_names)
        overlap_history = self.overlaps(result["theta_history"], names)

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(10, 5))

        for mu, name in enumerate(names):
            ax.plot(result["time"], overlap_history[:, mu], linewidth=2.0, label=f"Overlap with {name}")
        ax.set_title(title or f"Memory pattern overlaps (cued from corrupted '{result['cue']}')")
        ax.set_xlabel("Time (t)")
        ax.set_ylabel(r"Overlap $m^\mu(t)$")
        ax.legend(fontsize="small", ncol=3)
        ax.grid(True)

        if standalone:
            plt.show()


# ---------------------------------------------------------------------------
# Example usage (only runs when executed directly, not on import)
# ---------------------------------------------------------------------------

"""
if __name__ == "__main__":
    # All run parameters come from one explicit config -- there are no library-side
    # defaults for these, precisely so a notebook's CONFIG dict is the single source
    # of truth and can't silently diverge from what the library assumes. N and seed
    # are independent variables (swept across trials), so they're passed separately.
    CONFIG = dict(dt=0.02, T=55.0, frequency_std=0.03, phase_noise=0.01,
                  d=3, corruption_rate=0.2, mode="ode")

    # Reproduces the notebook's single-sequence recall demo. Pattern names are generated
    # from the sequence length, not hand-typed, since sequences can be any length.
    net = KuramotoNetwork(N=40, seed=10, **CONFIG)
    seq_len = 5
    names = [f"A_{k}" for k in range(1, seq_len + 1)]
    net.generate_patterns(names)
    net.add_sequence(names)
    result = net.simulate(cue_pattern=names[0])
    net.plot(result)
"""
