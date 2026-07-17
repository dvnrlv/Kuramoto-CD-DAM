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
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - allows the module to still import without numba
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrap(fn):
            return fn
        return wrap


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

    def mutate_pattern(self, base_name: str, new_name: str, corruption_rate: float, seed: Optional[int] = None) -> None:
        """Append a new stored pattern obtained by corrupting `base_name` (reuses `corrupt`,
        i.e. the same random-subset-of-spins flip 0 <-> pi used to build a cue). Unlike
        `generate_patterns`, the new pattern is correlated with its base rather than
        independent random noise -- this is the building block for a mutation-driven
        sequence (xi^{k+1} = mutate(xi^k))."""
        if new_name in self.name_to_index:
            raise ValueError(f"Pattern name already exists: {new_name}")
        new_phase = self.corrupt(base_name, corruption_rate, seed=seed)

        new_xi = np.cos(new_phase)
        self.phases = np.vstack([self.phases, new_phase[None, :]])
        self.xi = np.vstack([self.xi, new_xi[None, :]])
        self.name_to_index[new_name] = len(self.pattern_names)
        self.pattern_names.append(new_name)

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
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        names = [base_name]
        current = base_name
        for k in range(1, seq_len + 1):
            new_name = name_template.format(base=base_name, k=k)
            step_seed = int(rng.integers(0, 2**32 - 1))
            self.mutate_pattern(current, new_name, corruption_rate, seed=step_seed)
            names.append(new_name)
            current = new_name

        self.add_sequence(names)
        return names

    # -- sequence bookkeeping -------------------------------------------------

    def add_sequence(self, pattern_names_in_order: Sequence[str]) -> None:
        """Register the (single) stored sequence over already-generated pattern names."""
        missing = [n for n in pattern_names_in_order if n not in self.name_to_index]
        if missing:
            raise ValueError(f"Unknown pattern name(s): {missing}. Call generate_patterns first.")
        self.sequence = np.array(
            [self.name_to_index[n] for n in pattern_names_in_order], dtype=np.int64
        )

    def build_transition_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Flatten the stored sequence's mu -> mu+1 transitions into a (from, to) edge
        list. The final transition self-loops (xi^{P+1} == xi^P) rather than wrapping back
        to the first pattern, so the dynamics settle at the last pattern instead of
        cycling indefinitely."""
        if self.sequence is None:
            raise ValueError("No sequence registered; call add_sequence(...) first.")
        seq_idx = self.sequence
        edges_from = np.concatenate([seq_idx[:-1], seq_idx[-1:]])
        edges_to = np.concatenate([seq_idx[1:], seq_idx[-1:]])
        return edges_from.astype(np.int64), edges_to.astype(np.int64)

    # -- cueing ----------------------------------------------------------------

    def corrupt(self, pattern_name: str, corruption_rate: float, seed: Optional[int] = None) -> np.ndarray:
        """Return a corrupted copy (random subset of neurons flipped 0 <-> pi) of a
        pattern's phase vector, for use as an initial condition theta(0)."""
        idx = self.index_of(pattern_name)
        phase_pattern = self.phases[idx].copy()
        if corruption_rate <= 0:
            return phase_pattern

        rng = np.random.default_rng(seed if seed is not None else self.seed)
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
        library: Optional[PatternLibrary] = None,
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

        self.library = library or PatternLibrary(N=self.N, seed=self.seed)
        if self.library.N != self.N:
            raise ValueError("PatternLibrary.N must match KuramotoNetwork.N") # raise errors if mismatched number of neurons 
        self.omega = self._make_omega()

    # -- setup -----------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return self.num_steps_for(self.T)

    def num_steps_for(self, T: float) -> int:
        return int(T / self.dt)

    def time_vector(self, T: Optional[float] = None) -> np.ndarray:
        T = self.T if T is None else T
        return np.linspace(0.0, T, self.num_steps_for(T) + 1)

    def _make_omega(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, self.frequency_std, self.N)

    def generate_patterns(self, names: Sequence[str]) -> None:
        self.library.generate_patterns(names)

    def add_sequence(self, pattern_names_in_order: Sequence[str]) -> None:
        self.library.add_sequence(pattern_names_in_order)

    # -- simulation --------------------------------------------------------

    def simulate(
        self,
        mode: Optional[str] = None,
        cue_pattern: Optional[str] = None,
        corruption_rate: Optional[float] = None,
        seed: Optional[int] = None,
        T: Optional[float] = None,
    ) -> dict:
        """Run one trial.

        Parameters
        ----------
        mode : 'ode' (deterministic Forward Euler) or 'sde' (Euler-Maruyama with phase noise).
               Defaults to the network's configured `self.mode`.
        cue_pattern : name of the pattern to corrupt as the initial condition theta(0).
                      Defaults to the first pattern of the stored sequence.
        corruption_rate, seed : override config defaults for this trial.
        T : override the network's configured `self.T` for this trial only (`self.T` is left
            untouched). Useful when sweeping sequence length: the per-transition time budget
            shrinks as the chain gets longer unless `T` is scaled up to match, e.g.
            `net.simulate(..., T=dwell_time_per_transition * seq_len)`.

        Returns
        -------
        dict with 'time', 'theta_history', 'cue', plus (for convenience) 'overlaps'
        (overlap with every stored pattern over time).
        """
        mode = self.mode if mode is None else mode
        corruption_rate = self.corruption_rate if corruption_rate is None else corruption_rate
        seed = self.seed if seed is None else seed
        T = self.T if T is None else T

        if self.library.sequence is None:
            raise ValueError("No sequence registered; call add_sequence(...) first.")
        if cue_pattern is None:
            cue_pattern = self.library.pattern_names[self.library.sequence[0]]

        edges_from, edges_to = self.library.build_transition_edges()
        theta_init = self.library.corrupt(cue_pattern, corruption_rate=corruption_rate, seed=seed)
        num_steps = self.num_steps_for(T)

        if mode == "ode":
            theta_history = _simulate_euler(
                theta_init, self.omega, self.library.xi, edges_from, edges_to,
                self.d, self.dt, num_steps,
            )
        elif mode == "sde":
            theta_history = _simulate_maruyama(
                theta_init, self.omega, self.library.xi, edges_from, edges_to,
                self.d, self.dt, num_steps, self.phase_noise, seed,
            )
        else:
            raise ValueError("mode must be either 'ode' or 'sde'")

        result = {
            "time": self.time_vector(T),
            "theta_history": theta_history,
            "cue": cue_pattern,
        }
        result["overlaps"] = self.overlaps(theta_history)
        return result

    # -- diagnostics ---------------------------------------------------------

    def overlaps(self, theta_history: np.ndarray, pattern_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Global overlap m^mu(t) = (1/N) sum_i xi_i^mu cos(theta_i(t)) for chosen patterns
        (default: all stored patterns). Shape: (num_steps+1, num_patterns)."""
        if pattern_names is None:
            xi = self.library.xi
        else:
            xi = self.library.xi[[self.library.index_of(n) for n in pattern_names]]
        return np.cos(theta_history) @ xi.T / xi.shape[1]

    def recall_accuracy(self, result: dict) -> float:
        """Placeholder -- implement your own capacity/recall metric here.

        `result` is the dict returned by `simulate()` (has 'theta_history', 'time',
        'overlaps', ...); `self.overlaps(...)` and `self.library.sequence` are the
        building blocks you'll likely want.
        """
        raise NotImplementedError("recall_accuracy is left for you to implement.")

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
        from scipy.signal import find_peaks

        names = list(pattern_names) if pattern_names is not None else self.library.pattern_names
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
        names = list(pattern_names) if pattern_names is not None else self.library.pattern_names
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

    def compare_to_stored_sequence(self, retrieved: Sequence[str]) -> int:
        """Compare a decoded/retrieved sequence against the stored ground-truth sequence,
        position by position.

        Returns the 0-indexed step at which recall first fails (`retrieved[step] !=
        stored[step]`, or `retrieved` ran out early) -- i.e. how many leading steps were
        recalled correctly before the sequence broke. Returns `len(stored)` if `retrieved`
        matches the stored sequence exactly (perfect recall).
        """
        if self.library.sequence is None:
            raise ValueError("No sequence registered; call add_sequence(...) first.")
        stored = [self.library.pattern_names[i] for i in self.library.sequence]

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

        names = pattern_names if pattern_names is not None else self.library.pattern_names
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


