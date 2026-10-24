import numpy as np
from scipy.special import logsumexp

class HiddenMarkovModelLogSpace:
    def __init__(self, n_states, n_observations, seed=42):
        np.random.seed(seed)
        self.n_states = n_states
        self.n_observations = n_observations

        # Initialize probabilities randomly and normalize
        self.log_pi = np.log(np.random.rand(n_states))
        self.log_pi -= logsumexp(self.log_pi)

        self.log_A = np.log(np.random.rand(n_states, n_states))
        self.log_A -= logsumexp(self.log_A, axis=1, keepdims=True)

        self.log_B = np.log(np.random.rand(n_states, n_observations))
        self.log_B -= logsumexp(self.log_B, axis=1, keepdims=True)

    def fit(self, observations, n_iters=100):
        T = len(observations)

        for iteration in range(n_iters):
            # Log Forward
            log_alpha = np.zeros((T, self.n_states))

            log_alpha[0] = self.log_pi + self.log_B[:, observations[0]]

            for t in range(1, T):
                for j in range(self.n_states):
                    log_alpha[t, j] = logsumexp(log_alpha[t-1] + self.log_A[:, j]) + self.log_B[j, observations[t]]

            # Log Backward
            log_beta = np.zeros((T, self.n_states))
            log_beta[T-1] = 0.0 # log(1)

            for t in range(T-2, -1, -1):
                for i in range(self.n_states):
                    log_beta[t, i] = logsumexp(self.log_A[i, :] + self.log_B[:, observations[t+1]] + log_beta[t+1, :])

            # Compute log_gamma and log_xi
            log_gamma = np.zeros((T, self.n_states))
            log_xi = np.zeros((T-1, self.n_states, self.n_states))

            for t in range(T):
                log_gamma[t] = log_alpha[t] + log_beta[t]
                log_gamma[t] -= logsumexp(log_gamma[t])

            for t in range(T-1):
                for i in range(self.n_states):
                    log_xi[t, i, :] = log_alpha[t, i] + self.log_A[i, :] + self.log_B[:, observations[t+1]] + log_beta[t+1, :]
                log_xi[t] -= logsumexp(log_xi[t])

            # M-step
            self.log_pi = log_gamma[0]

            self.log_A = logsumexp(log_xi, axis=0) - logsumexp(log_gamma[:-1], axis=0).reshape(-1, 1)
            self.log_A -= logsumexp(self.log_A, axis=1, keepdims=True) # Ensure row normalization

            for k in range(self.n_observations):
                mask = (observations == k)
                if np.any(mask):
                    self.log_B[:, k] = logsumexp(log_gamma[mask], axis=0) - logsumexp(log_gamma, axis=0)
                else:
                    self.log_B[:, k] = -np.inf
            self.log_B -= logsumexp(self.log_B, axis=1, keepdims=True)

if __name__ == "__main__":
    print("Testing Hidden Markov Model (HMM) component in Log-Space...")

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

    hmm = HiddenMarkovModelLogSpace(n_states=2, n_observations=3, seed=42)
    hmm.fit(obs, n_iters=100)

    # Convert log probs back to probs for comparison
    estimated_B = np.exp(hmm.log_B)

    match_0 = np.linalg.norm(estimated_B[0] - true_B[0]) + np.linalg.norm(estimated_B[1] - true_B[1])
    match_1 = np.linalg.norm(estimated_B[0] - true_B[1]) + np.linalg.norm(estimated_B[1] - true_B[0])

    print(f"Match 0 distance: {match_0:.4f}")
    print(f"Match 1 distance: {match_1:.4f}")

    if match_0 < 0.5 or match_1 < 0.5:
        print("Successfully learned parameters representing the underlying states!")
    else:
        print("Failed to converge to expected matrices.")
        exit(1)
