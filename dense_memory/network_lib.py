# This is source code for function used for retrieval of patterns from Hopfield/ Oscillator Networks for binary patterns

import numpy as np
from numba import njit


def hebbian_learning(patterns):
    P, N = patterns.shape  # P = number of patterns, N = number of oscillators
    K = np.zeros((N, N))
    
    # Apply the Hebbian learning rule
    for p in range(P):
        pattern = patterns[p, :].reshape(N, 1)
        K += pattern @ pattern.T  # Outer product of the pattern with itself
    
    K /= P
    return K

def generate_J(N, patterns):

    #why fourth order? what is the benefit of using fourth order interactions?

    #why normalised by N^3 instead of P?

    # Expand dimensions for broadcasting
    patterns_i = patterns[:, :, np.newaxis, np.newaxis, np.newaxis]  # (M, N, 1, 1, 1)
    patterns_j = patterns[:, np.newaxis, :, np.newaxis, np.newaxis]  # (M, 1, N, 1, 1)
    patterns_k = patterns[:, np.newaxis, np.newaxis, :, np.newaxis]  # (M, 1, 1, N, 1)
    patterns_l = patterns[:, np.newaxis, np.newaxis, np.newaxis, :]  # (M, 1, 1, 1, N)

    # Compute the outer product and sum over all patterns
    J = np.sum(patterns_i * patterns_j * patterns_k * patterns_l, axis=0) / (N ** 3)
    return J

def overlap_phase_patterns(phase, patterns):
    N_patterns = patterns.shape[0]
    N = phase.shape[0]
    exp_phases = np.exp(1j * phase) 
    overlap_patterns = np.abs(np.dot(exp_phases, patterns.T)) / N
    return overlap_patterns


def overlap_phases(phase1, phase2):
    phase1 = np.asarray(phase1, dtype=np.float64)
    phase2 = np.asarray(phase2, dtype=np.float64)
    return np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
def overlap_patterns(pattern1, pattern2):
    pattern1 = np.asarray(pattern1, dtype=np.float64)
    pattern2 = np.asarray(pattern2, dtype=np.float64)
    return np.abs(np.mean(pattern1 * pattern2))
def overlap_phase_pattern(phase, pattern):
    phase = np.asarray(phase, dtype=np.float64)
    pattern = np.asarray(pattern, dtype=np.float64)
    exp_phase = np.exp(1j * phase)
    return np.abs(np.mean(exp_phase * pattern))

#what is the purpose of different overlap functions?

def generateOrthogonalPatterns(N, Np):
    # Check if N is a power of 2
    if N & (N - 1) != 0:
        raise ValueError("N must be a power of 2")
    
    n = int(np.log2(N))
    sequences = np.array([[1, 1], [1, -1]])
    for _ in range(1, n):
        sequences = np.vstack([
            np.hstack([sequences, sequences]),
            np.hstack([sequences, -sequences])
        ])

    patterns = []
    for i in range(Np):
        pattern_row = (sequences[i % len(sequences)])
        patterns.append(pattern_row)

    return np.array(patterns)


@njit(parallel=True)
def compute_J_norms(patterns, N, P):
    norms_J = np.zeros(N)
    for i in prange(N):
        acc = 0.0
        for j in range(N):
            tmp = 0.0
            for m in range(P):
                tmp += patterns[m, i] * patterns[m, j]
            J_ij = tmp 
            
            acc += abs(J_ij)
        norms_J[i] = acc / N
    norms_J[norms_J == 0] = 1.0  # Avoid division by zero
    return norms_J

@njit(parallel=True)
def compute_K_norms(patterns, N, P):
    norms_K = np.zeros(N)
    for i in prange(N):
        acc = 0.0
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    tmp = 0.0
                    for m in range(P):
                        tmp += patterns[m, i] * patterns[m, j] * patterns[m, k] * patterns[m, l]
                    K_ijkl = tmp 
                    acc += abs(K_ijkl)
        norms_K[i] = acc / (N**3)
    norms_K[norms_K == 0] = 1.0  # Avoid division by zero
    return norms_K

@njit(parallel=True)
def compute_K_norms_sampled(patterns, N, P, M=10000):
    norms_K = np.zeros(N)
    for i in prange(N):
        acc = 0.0
        for s in range(M):
            j = np.random.randint(0, N)
            k = np.random.randint(0, N)
            l = np.random.randint(0, N)
            K_ijkl = np.sum(patterns[:, i] * patterns[:, j] * patterns[:, k] * patterns[:, l]) 
            acc += abs(K_ijkl)
        norms_K[i] = acc / M 
    norms_K[norms_K == 0] = 1.0
    return norms_K

###################################################################################
# Classes for Networks
###################################################################################
# Hopfield
class HopfieldNetwork:
    def __init__(self, num_oscillators):
        self.N = num_oscillators
        self.weights = np.zeros((self.N, self.N))

    def train(self, patterns):
        self.P = patterns.shape[0]
        self.weights += np.einsum('mi,mj->ij', patterns, patterns)/ self.P
        np.fill_diagonal(self.weights, 0)

    def energy(self, state):
        return -0.5 * np.dot(state, np.dot(self.weights, state))/ self.N

    def update(self, state, synchronous=True):
        if synchronous:
            new_state = np.sign(np.dot(self.weights, state))
            new_state[new_state == 0] = 1
            return new_state

        else:
            new_state = state.copy()
            i = np.random.randint(0, self.N)
            new_state[i] = np.sign(np.dot(self.weights[i], state)/self.N)
            if new_state[i] == 0:
                new_state[i] = 1 
            return new_state

    def recall(self, state, max_iterations=10, synchronous=True):

        history = [state.copy()] 
        for _ in range(max_iterations):
            state = self.update(state, synchronous)
            history.append(state.copy())

            if synchronous and np.array_equal(history[-1], history[-2]):
                #print('Converged')
                break
        return np.array(history)
# higher order Hopfield Network
class HigherOrderHopfieldNetwork:
    def __init__(self, num_oscillators, n=4):
        self.N = num_oscillators
        self.n = n  # Order of the polynomial function

    def rectified_polynomial(self, x):
        return np.where(x >= 0, x**self.n, 0)

    def train(self, patterns):
        self.patterns = patterns
        self.P = patterns.shape[0]

    def energy(self, state):
        input_sums = np.dot(self.patterns, state)
        energies = self.rectified_polynomial(input_sums)  # shape: (P,)
        return -np.sum(energies)
    
    def update_asynchronus(self, state):
        new_state = state.copy()
        i = np.random.randint(self.N)

        input_sum = np.sum(state * self.patterns, axis=1) - self.patterns[:,i] *state[i]
        
        F_positive = self.rectified_polynomial(self.patterns[:, i] + input_sum)
        F_negative = self.rectified_polynomial(-self.patterns[:, i] + input_sum)

        new_state[i] = np.sign(np.sum(F_positive - F_negative))

        if new_state[i] == 0:
            new_state[i] = 1  

        return new_state
    
    def update(self, state, synchronous=True):
        new_state = state.copy()
        if synchronous:
            for i in range(self.N):
                input_sum = np.sum(state * self.patterns, axis=1) - self.patterns[:, i] * state[i]

                F_positive = self.rectified_polynomial(self.patterns[:, i] + input_sum)
                F_negative = self.rectified_polynomial(-self.patterns[:, i] + input_sum)

                new_state[i] = np.sign(np.sum(F_positive - F_negative))

                if new_state[i] == 0:
                    new_state[i] = 1

        else:
            i = np.random.randint(self.N)

            input_sum = np.sum(state * self.patterns, axis=1) - self.patterns[:,i] *state[i]
            
            F_positive = self.rectified_polynomial(self.patterns[:, i] + input_sum)
            F_negative = self.rectified_polynomial(-self.patterns[:, i] + input_sum)

            new_state[i] = np.sign(np.sum(F_positive - F_negative))

            if new_state[i] == 0:
                new_state[i] = 1  
        return new_state

    def recall(self, state, max_iterations=10, synchronous=True):
        history = [state.copy()]
        for _ in range(max_iterations):
            state = self.update(state, synchronous=synchronous)
            history.append(state.copy())

            if synchronous and np.array_equal(history[-1], history[-2]):
                #print('Converged')
                break

        return np.array(history)

import numpy as np
from numba import njit, prange


@njit
def energy_HOkuramoto_orderparam(state, patterns, J, K, N, P, norms_J, norms_K):
    energy = 0.0

    # Compute order parameters
    S_plus = np.zeros(P, dtype=np.complex128)
    for m in range(P):
        for j in range(N):
            S_plus[m] += patterns[m, j] * np.exp(1j * state[j])
    S_minus = np.conj(S_plus)

    # Energy contribution from each oscillator
    for i in range(N):
        energy_second = 0.0
        energy_higher = 0.0

        for m in range(P):
            z_i = patterns[m, i] * np.exp(-1j * state[i])
            energy_second += np.real(z_i * S_plus[m])
            energy_higher += np.real(z_i * (S_plus[m]**2) * S_minus[m])

        energy_second *= J / (2*N * norms_J[i])
        energy_higher *= K / (24 * N**3 * norms_K[i])

        energy += -(energy_second + energy_higher)
    return energy

@njit
def fast_update_HOkuramoto_orderparam(patterns, state, dt, J, K, omega, N, P, norms_J, norms_K, D):
    new_state = np.empty_like(state)

    # Compute order parameters S_+ for each pattern μ
    S_plus = np.zeros(P, dtype=np.complex128)
    for m in range(P):
        for j in range(N):
            S_plus[m] += patterns[m, j] * np.exp(1j * state[j])
    S_minus = np.conj(S_plus)

    sqrt_2Ddt = np.sqrt(2 * D * dt)

    # Now update each oscillator i
    for i in range(N):
        # Second order term
        sum_second = 0.0
        for m in range(P):
            sum_second += patterns[m, i] * np.imag(S_plus[m] * np.exp(-1j * state[i]))
        sum_second *= J / (N*norms_J[i])

        # Higher order term
        sum_higher = 0.0
        for m in range(P):
            order_term = (S_plus[m]**2) * S_minus[m] * np.exp(-1j * state[i])
            sum_higher += patterns[m, i] * np.imag(order_term)
        sum_higher *= K / (N**3 * norms_K[i]) /6

        # Langevin noise
        noise = sqrt_2Ddt * np.random.normal()

        # Total force update
        total_force = sum_second + sum_higher
        new_state[i] = state[i] + dt * (total_force + omega[i]) + noise

    return new_state

class HigherOrderKuramoto:
    def __init__(self, num_oscillators, J=1.0, K=1.0, omega=None, D=0.0):
        self.N = num_oscillators
        self.J = J
        self.K = K
        self.D = D  # thermal noise intensity
        self.omega = omega if omega is not None else np.zeros(self.N)

    def train(self, patterns, norm=True):
        N = self.N
        P = patterns.shape[0]
        self.patterns = patterns
        self.P = P
        if norm:
            self.norms_J = compute_J_norms(patterns, N, P)
            self.norms_K = compute_K_norms_sampled(patterns, N, P)
        else:
            self.norms_J = np.ones(N)
            self.norms_K = np.ones(N)

    def recall(self, state, T=10, dt=0.01):
        time_steps = int(T / dt)
        history = [state.copy()]
        times = [0]
        for t in range(time_steps):
            state = fast_update_HOkuramoto_orderparam(
                self.patterns, state, dt,
                self.J, self.K, self.omega, self.N, self.P, self.norms_J, self.norms_K,
                self.D
            )
            history.append(state.copy())
            times.append((t+1) * dt)
        return np.array(times), np.array(history) % (2*np.pi)
    
    def energy(self, state):
        return energy_HOkuramoto_orderparam(
            state, self.patterns, self.J, self.K, self.N, self.P,
            self.norms_J, self.norms_K
        )

from numba import njit
@njit
def fast_update_Kuramoto2nd_scaled(state, patterns, alpha, dt, N, P):
    S_plus = np.zeros(P, dtype=np.complex64)
    for m in range(P):
        for j in range(N):
            S_plus[m] += patterns[m, j] * np.exp(1j * state[j])

    S2_plus = 0.0 + 0.0j
    for j in range(N):
        S2_plus += np.exp(2j * state[j])

    new_state = np.empty_like(state)
    for i in range(N):
        total_force = 0.0
        for m in range(P):
            total_force += patterns[m, i] * np.imag(S_plus[m] * np.exp(-1j * state[i])) / N
        total_force += (alpha / N) * np.imag(S2_plus * np.exp(-2j * state[i]))
        new_state[i] = state[i] + dt * total_force
    return new_state

@njit
def energy_Kuramoto2nd_scaled(state, patterns, alpha):
    energy = 0.0
    n = state.shape[0]
    p = patterns.shape[0]
    for i in range(n):
        for j in range(n):
            J_contrib = 0.0
            for m in range(p):
                J_contrib += patterns[m, i] * patterns[m, j]
            energy += -0.5 * J_contrib * np.cos(state[j] - state[i]) / n
            energy += -0.25 * alpha * np.cos(2 * (state[j] - state[i])) / n
    return energy
@njit
def energy_Kuramoto2nd_orderparam(state, patterns, alpha):
    n = state.shape[0]
    p = patterns.shape[0]

    # Compute order parameters
    S_plus = np.zeros(p, dtype=np.complex128)
    for m in range(p):
        for j in range(n):
            S_plus[m] += patterns[m, j] * np.exp(1j * state[j])

    S2_plus = 0.0 + 0.0j
    for m in range(p):
        for j in range(n):
            S_plus[m] += patterns[m, j] * np.exp(1j * state[j])
    for j in range(n):
        S2_plus += np.exp(2j * state[j])

    energy = 0.0
    for i in range(n):
        z_i = np.exp(-1j * state[i])
        e2_i = np.exp(-2j * state[i])

        # Second order energy
        E2_i = 0.0
        for m in range(p):
            E2_i += np.real(patterns[m, i] * z_i * S_plus[m])
        E2_i *= 1 / (2*n)

        # Second harmonic energy
        E4_i = 0.25 * alpha / n * np.real(e2_i * S2_plus)

        energy += -(E2_i + E4_i)
    return energy


class KuramotoNetwork2ndOrder:
    def __init__(self, num_oscillators, alpha=0.5):
        self.N = num_oscillators
        self.alpha = alpha  

    def train(self, patterns):
        self.patterns = patterns
        self.P = patterns.shape[0]
        P = self.P
        N = self.N


    def recall(self, state, T=10, dt=0.01):
        time_steps = int(T / dt)
        times = np.linspace(0, T, time_steps + 1)
        history = np.empty((time_steps + 1, self.N))
        history[0] = state.copy()

        for t in range(1, time_steps + 1):
            state = fast_update_Kuramoto2nd_scaled(
                state, self.patterns, self.alpha, dt, self.N, self.P)
            history[t] = state
        return times, history % (2*np.pi)

    def energy(self, state):
        return energy_Kuramoto2nd_orderparam(state, self.patterns, self.alpha)

# Continuous Hopfield Network 
class HopfieldPRL:
    def __init__(self, num_neurons, beta=2):
        self.N = num_neurons
        self.k = 4
        self.beta = beta
        self.weights = np.zeros((self.N, self.N, self.N, self.N))

    def train(self, patterns):
        self.P = patterns.shape[0]
        self.weights += np.einsum('mi,mj,mk,ml->ijkl', patterns, patterns, patterns, patterns)
        self.weights /= self.P

    def _activation(self, x):
        return np.tanh(x / self.beta)

    def _dynamics(self, x):
        s = self._activation(x)
        dxdt = np.einsum('i,j,k,lijk->l', s, s, s, self.weights)/ (self.N**3) #-x
        return dxdt
    
    def energy(self, x):
        s = self._activation(x)
        energy = -0.25 * np.einsum('i,j,k,lijk->', s, s, s, self.weights)/(self.N**3*self.P)
        return energy

    def recall(self, x0, T=5.0, dt=0.05):
        steps = int(T / dt)
        x = x0.copy()
        history = []
        times = []

        for step in range(steps + 1):
            history.append(self._activation(x.copy()))
            times.append(step * dt)
            dxdt = self._dynamics(x)
            x += dt * dxdt

        return np.array(times), np.array(history)