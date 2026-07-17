"""
run_experiment.py

Experiment pipeline built on top of networks.py:
  1. Pick a network type, N (units), P (stored patterns).
  2. Generate P random patterns and train the network on them.
  3. Corrupt one stored pattern to build an initial state.
  4. Run recall/relaxation from that corrupted state.
  5. Compute overlap with every stored pattern at each step and plot it.

Usage:
    python run_experiment.py --network hopfield --N 200 --P 6
    python run_experiment.py --network kuramoto_low --N 200 --P 6
    python run_experiment.py --network kuramoto_high --N 100 --P 5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from implementation.CDM2026.dense_memory.network_lib import *

# --------------------------------------------------------------------------
# Pattern generation / corruption
# --------------------------------------------------------------------------


def generate_random_patterns(P, N, rng):
    return rng.choice([-1.0, 1.0], size=(P, N))


def corrupt_binary_pattern(pattern, flip_fraction, rng):
    """Flip a random fraction of entries of a +-1 pattern."""
    corrupted = pattern.copy()
    n_flip = int(round(flip_fraction * len(pattern)))
    idx = rng.choice(len(pattern), size=n_flip, replace=False)
    corrupted[idx] *= -1
    return corrupted


def pattern_to_phase(pattern, noise_std, rng):
    """Map a +-1 pattern to phases (0 or pi) and add Gaussian phase noise."""
    base_phase = np.where(pattern > 0, 0.0, np.pi)
    noise = rng.normal(0, noise_std, size=pattern.shape)
    return (base_phase + noise) % (2 * np.pi)


def adaptive_recall(net, init_state, network_type, max_iterations, T, dt,
                    tol=1e-8, k_consec=5, div_tol=1.0, max_steps=100000,
                    chunk_T=1.0):
    """Adaptive recall: iterate until convergence/divergence criteria are met.

    For discrete networks, uses `net.update()` repeatedly. For continuous
    networks (listed in `CONTINUOUS_NETWORKS`) it runs `net.recall` in chunks
    of duration `chunk_T` and inspects recent changes.

    Returns (steps, history) where steps is either np.arange(len(history))
    for discrete or an array of times for continuous.
    """
    is_continuous = network_type in CONTINUOUS_NETWORKS

    if is_continuous:
        times_all = []
        history_all = []
        current_state = init_state.copy()
        total_time = 0.0
        steps = 0
        while True:
            times_chunk, history_chunk = net.recall(current_state, T=chunk_T, dt=dt)
            if len(history_all) == 0:
                times_all.extend(list(times_chunk))
                history_all.extend([h.copy() for h in history_chunk])
            else:
                times_all.extend([total_time + t for t in times_chunk[1:]])
                history_all.extend([h.copy() for h in history_chunk[1:]])

            # convergence/divergence checks on most recent k_consec changes
            if len(history_all) >= k_consec + 1:
                recent_changes = [np.max(np.abs(history_all[-i] - history_all[-i-1])) for i in range(1, k_consec+1)]
                if all(c < tol for c in recent_changes):
                    break
                if np.mean(recent_changes) > div_tol:
                    break

            current_state = history_chunk[-1].copy()
            total_time += times_chunk[-1]
            steps += len(history_chunk) - 1
            if steps > max_steps:
                break

        return np.array(times_all), np.array(history_all)

    else:
        history = [init_state.copy()]
        state = init_state.copy()
        consec = 0
        steps = 0
        while True:
            state = net.update(state, synchronous=True)
            history.append(state.copy())
            steps += 1
            change = np.max(np.abs(history[-1] - history[-2]))
            if change == 0:
                break
            if change < tol:
                consec += 1
                if consec >= k_consec:
                    break
            else:
                consec = 0
            if steps >= max_steps:
                break

        return np.arange(len(history)), np.array(history)


# --------------------------------------------------------------------------
# Experiment orchestration
# --------------------------------------------------------------------------


def run_experiment(network_type, N, P, max_iterations, corruption=0.25, target_idx=0,
                    T=10.0, dt=0.02, seed=0):
    rng = np.random.default_rng(seed)
    patterns = generate_random_patterns(P, N, rng)
    target = patterns[target_idx]

    if network_type == "hopfield":
        net = HopfieldNetwork(N)
        net.train(patterns)
        init_state = corrupt_binary_pattern(target, corruption, rng)
        steps, history = adaptive_recall(net, init_state, network_type, max_iterations, T, dt)
        overlaps = np.array([[overlap_patterns(s, p) for p in patterns] for s in history])
        x_label = "iteration"

    elif network_type == "hopfield_higher":
        net = HigherOrderHopfieldNetwork(N)
        net.train(patterns)
        init_state = corrupt_binary_pattern(target, corruption, rng)
        steps, history = adaptive_recall(net, init_state, network_type, max_iterations, T, dt)
        overlaps = np.array([[overlap_patterns(s, p) for p in patterns] for s in history])
        x_label = "iteration"

    elif network_type == "kuramoto_low":
        net = KuramotoNetwork2ndOrder(N, alpha=0.0)
        net.train(patterns)
        init_state = pattern_to_phase(target, noise_std=corruption * np.pi, rng=rng)
        steps, history = adaptive_recall(net, init_state, network_type, max_iterations, T, dt)
        overlaps = np.array([overlap_phase_patterns(s, patterns) for s in history])
        x_label = "time"

    elif network_type == "kuramoto_high":
        net = HigherOrderKuramoto(N, J=1.0, K=1.0, D=0.0)
        net.train(patterns, norm=True)
        init_state = pattern_to_phase(target, noise_std=corruption * np.pi, rng=rng)
        steps, history = adaptive_recall(net, init_state, network_type, max_iterations, T, dt)
        overlaps = np.array([overlap_phase_patterns(s, patterns) for s in history])
        x_label = "time"

    elif network_type == "hopfield_prl":
        net = HopfieldPRL(N)
        net.train(patterns)
        init_state = corrupt_binary_pattern(target, corruption, rng)
        steps, history = adaptive_recall(net, init_state, network_type, max_iterations, T, dt)
        overlaps = np.array([[overlap_patterns(s, p) for p in patterns] for s in history])
        x_label = "time"

    else:
        raise ValueError(
            "network_type must be one of: 'hopfield', 'hopfield_higher', 'kuramoto_low', 'kuramoto_high', 'hopfield_prl'"
        )

    return steps, overlaps, x_label, target_idx


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def plot_overlaps(steps, overlaps, x_label, target_idx, network_type, save_path=None):
    import matplotlib.pyplot as plt

    P = overlaps.shape[1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in range(P):
        style = "-" if p == target_idx else "--"
        lw = 2.5 if p == target_idx else 1.2
        alpha = 1.0 if p == target_idx else 0.6
        label = f"pattern {p}" + (" (target)" if p == target_idx else "")
        ax.plot(steps, overlaps[:, p], style, linewidth=lw, alpha=alpha, label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel("overlap with stored pattern")
    ax.set_title(f"Recall dynamics: {network_type}")
    ax.set_ylim(-0.05, 1.05) if overlaps.min() >= -1e-6 else ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# Energy landscape plotting removed per user request.

NETWORK_CHOICES = {
    "1": "hopfield",
    "2": "hopfield_higher",
    "3": "kuramoto_low",
    "4": "kuramoto_high",
    "5": "hopfield_prl",
}
 
# Networks whose recall() takes T/dt (continuous time) vs max_iterations (discrete steps)
CONTINUOUS_NETWORKS = {"kuramoto_low", "kuramoto_high", "hopfield_prl"}
 
 
def ask(prompt, default, cast=str):
    """Prompt the user for a value; press Enter to accept the default."""
    raw = input(f"{prompt} [default {default}]: ").strip()
    if raw == "":
        return default
    return cast(raw)
 
 
def choose_network():
    print("Which network do you want to use?")
    print("  1) hopfield         - standard 2nd-order Hopfield (binary, discrete)")
    print("  2) hopfield_higher  - higher-order Hopfield (binary, discrete)")
    print("  3) kuramoto_low     - Kuramoto, pairwise coupling only (phase)")
    print("  4) kuramoto_high    - Kuramoto, 2nd + 4th order coupling (phase)")
    print("  5) hopfield_prl     - continuous 4th-order dense associative memory")
    while True:
        choice = input("Enter a number [default 1]: ").strip() or "1"
        if choice in NETWORK_CHOICES:
            return NETWORK_CHOICES[choice]
        print("Please enter one of: 1, 2, 3, 4, 5")
 
 
def main():
    print("=== Associative Memory Recall Demo ===\n")
 
    network_type = choose_network()
    N = 2**ask("Number of neurons (power of 2)", 8, int)
    P = ask("Number of stored patterns (P)", 6, int)
    corruption = ask("Corruption level of initial state (0-1)", 0.25, float)
    seed = ask("Random seed", 0, int)
 
    if network_type in CONTINUOUS_NETWORKS:
        T = 10
        dt = 0.02
        max_iterations = 30  # unused for these networks
    else:
        max_iterations = ask("Max iterations (discrete Hopfield networks)", 30, int)
        T, dt = 10.0, 0.02  # unused for these networks
 
    print("\nRunning experiment...")
    steps, overlaps, x_label, target_idx = run_experiment(
        network_type=network_type, N=N, P=P,
        corruption=corruption, target_idx=0,
        T=T, dt=dt, seed=seed,
        max_iterations=max_iterations,
    )
    # Show the overlaps plot interactively instead of saving to file
    fig = plot_overlaps(steps, overlaps, x_label, target_idx, network_type, save_path=None)
    
    print(f"Final overlap with target pattern: {overlaps[-1, target_idx]:.3f}")
    plt.show()


if __name__ == "__main__":
    main()