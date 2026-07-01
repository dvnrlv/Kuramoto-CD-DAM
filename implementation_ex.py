import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from context_tools import (
	HopfieldNetwork,
	HigherOrderKuramoto,
	generateOrthogonalPatterns,
	overlap_phase_patterns,
	overlap_patterns,
)


def corrupt_phase(phase, frac):
	"""Flip phase by pi at random fraction of sites."""
	new = phase.copy()
	nflip = max(1, int(len(phase) * frac))
	idx = np.random.choice(len(phase), nflip, replace=False)
	new[idx] = (new[idx] + np.pi) % (2 * np.pi)
	return new


def phase_to_binary(phase):
	# map phase -> ±1 using sign of cos(phase)
	return np.where(np.cos(phase) >= 0, 1, -1)


def run_higher_order_kuramoto_example():
	np.random.seed(1)
	N = 32
	P = 3

	# generate orthogonal-like ±1 patterns
	patterns = generateOrthogonalPatterns(N, P)

	# display stored patterns (print and image)
	print('Stored patterns (rows = patterns, cols = neurons):')
	print(patterns)

	# create and train network with noise present (original example)
	net = HigherOrderKuramoto(N, J=2.0, K=1.0, D=0.01)
	net.train(patterns, norm=True)

	# initialize phases aligned to pattern 0 (0 for +1, pi for -1)
	base_phase = np.where(patterns[0] == 1, 0.0, np.pi)
	init_phase = corrupt_phase(base_phase, frac=0.2)  # 20% corruption

	# run recall (original T and dt)
	times, history = net.recall(init_phase, T=5.0, dt=0.01)

	# compute overlaps over time (time x P)
	overlaps = np.array([overlap_phase_patterns(history[t], patterns) for t in range(len(history))])

	# prepare images: stored patterns, initial attempt, final attempt
	initial_bin = phase_to_binary(init_phase)
	final_bin = phase_to_binary(history[-1])

	# plot patterns and attempts
	fig, axes = plt.subplots(1, 3, figsize=(12, 4))
	axes[0].imshow(patterns, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
	axes[0].set_title('Stored patterns (P x N)')
	axes[1].imshow(initial_bin[np.newaxis, :], aspect='auto', cmap='bwr', vmin=-1, vmax=1)
	axes[1].set_title('Initial attempt (mapped)')
	axes[2].imshow(final_bin[np.newaxis, :], aspect='auto', cmap='bwr', vmin=-1, vmax=1)
	axes[2].set_title('Final attempt (mapped)')
	for a in axes:
		a.set_ylabel('pattern / attempt')
		a.set_xlabel('neuron index')

	plt.tight_layout()

	# plot overlaps
	plt.figure(figsize=(6, 4))
	for mu in range(P):
		plt.plot(times, overlaps[:, mu], label=f'pattern {mu}')
	plt.xlabel('time')
	plt.ylabel('overlap |R^mu|')
	plt.title('Overlap convergence')
	plt.legend()
	plt.tight_layout()

	print('Initial overlaps:', overlaps[0])
	print('Final overlaps:', overlaps[-1])

	# check binary overlap with target pattern (should be 1 for exact retrieval)
	final_bin_overlap = overlap_patterns(final_bin, patterns[0])
	print('Final binary overlap with target pattern 0:', final_bin_overlap)

	plt.show()


if __name__ == '__main__':
	run_higher_order_kuramoto_example()


def flip_bits(pattern, frac):
	s = pattern.copy()
	nflip = max(1, int(len(s) * frac))
	idx = np.random.choice(len(s), nflip, replace=False)
	s[idx] *= -1
	return s


def main():
	np.random.seed(0)
	N = 16        # number of neurons (must be power of 2 for generateOrthogonalPatterns)
	P = 3         # number of stored patterns

	patterns = generateOrthogonalPatterns(N, P)  # shape (P, N), entries ±1

	net = HopfieldNetwork(N)
	net.train(patterns)

	target = patterns[0].copy()
	init_state = flip_bits(target, frac=0.2)   # corrupt 20% of bits

	print("Initial overlaps:", [float(overlap_patterns(init_state, patterns[i])) for i in range(P)])
	print("Initial energy:", net.energy(init_state))

	history = net.recall(init_state, max_iterations=50, synchronous=True)
	final = history[-1]

	print("Converged in steps:", len(history) - 1)
	print("Final overlaps:", [float(overlap_patterns(final, patterns[i])) for i in range(P)])
	print("Final energy:", net.energy(final))


if __name__ == "__main__":
	main()



