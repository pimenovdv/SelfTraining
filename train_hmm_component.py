import numpy as np

class HiddenMarkovModel:
    def __init__(self, n_states, n_observations, seed=42):
        np.random.seed(seed)
        self.n_states = n_states
        self.n_observations = n_observations

        # Initialize probabilities randomly and normalize
        self.pi = np.random.rand(n_states)
        self.pi /= np.sum(self.pi)

        self.A = np.random.rand(n_states, n_states)
        self.A /= np.sum(self.A, axis=1, keepdims=True)

        self.B = np.random.rand(n_states, n_observations)
        self.B /= np.sum(self.B, axis=1, keepdims=True)

    def fit(self, observations, n_iters=100):
        T = len(observations)

        for iteration in range(n_iters):
            # Scaled Forward
            alpha = np.zeros((T, self.n_states))
            c = np.zeros(T)

            alpha[0] = self.pi * self.B[:, observations[0]]
            c[0] = np.sum(alpha[0])
            if c[0] == 0: c[0] = 1e-10
            alpha[0] /= c[0]

            for t in range(1, T):
                alpha[t] = np.dot(alpha[t-1], self.A) * self.B[:, observations[t]]
                c[t] = np.sum(alpha[t])
                if c[t] == 0: c[t] = 1e-10
                alpha[t] /= c[t]

            # Scaled Backward
            beta = np.zeros((T, self.n_states))
            beta[T-1] = 1.0

            for t in range(T-2, -1, -1):
                beta[t] = np.dot(self.A, (self.B[:, observations[t+1]] * beta[t+1]))
                beta[t] /= c[t+1]

            # Compute gamma and xi
            gamma = np.zeros((T, self.n_states))
            xi = np.zeros((T-1, self.n_states, self.n_states))

            for t in range(T):
                gamma[t] = alpha[t] * beta[t]
                norm = np.sum(gamma[t])
                if norm == 0: norm = 1e-10
                gamma[t] /= norm

            for t in range(T-1):
                for i in range(self.n_states):
                    xi[t, i, :] = alpha[t, i] * self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1, :]
                norm = np.sum(xi[t])
                if norm == 0: norm = 1e-10
                xi[t] /= norm

            # M-step
            self.pi = gamma[0]

            self.A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0).reshape(-1, 1)
            self.A /= np.sum(self.A, axis=1, keepdims=True)

            for k in range(self.n_observations):
                mask = (observations == k)
                self.B[:, k] = np.sum(gamma[mask], axis=0) / np.sum(gamma, axis=0)
            self.B /= np.sum(self.B, axis=1, keepdims=True)

if __name__ == "__main__":
    print("Testing Hidden Markov Model (HMM) component...")

    true_pi = np.array([0.6, 0.4])
    true_A = np.array([[0.8, 0.2],
                       [0.3, 0.7]])
    true_B = np.array([[0.8, 0.1, 0.1],
                       [0.1, 0.4, 0.5]])

    np.random.seed(42)
    seq_len = 1000
    states = [np.random.choice(2, p=true_pi)]
    obs = [np.random.choice(3, p=true_B[states[0]])]

    for _ in range(1, seq_len):
        next_state = np.random.choice(2, p=true_A[states[-1]])
        states.append(next_state)
        obs.append(np.random.choice(3, p=true_B[next_state]))

    obs = np.array(obs)

    hmm = HiddenMarkovModel(n_states=2, n_observations=3, seed=42)
    hmm.fit(obs, n_iters=100)

    match_0 = np.linalg.norm(hmm.B[0] - true_B[0]) + np.linalg.norm(hmm.B[1] - true_B[1])
    match_1 = np.linalg.norm(hmm.B[0] - true_B[1]) + np.linalg.norm(hmm.B[1] - true_B[0])

    print(f"Match 0 distance: {match_0:.4f}")
    print(f"Match 1 distance: {match_1:.4f}")

    if match_0 < 0.5 or match_1 < 0.5:
        print("Successfully learned parameters representing the underlying states!")
    else:
        print("Failed to converge to expected matrices.")
        exit(1)
