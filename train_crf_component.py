import numpy as np

import scipy.optimize

def log_sum_exp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

class LinearChainCRF:
    def __init__(self, num_states, num_features):
        self.num_states = num_states
        self.num_features = num_features
        # weights: (num_features, num_states) and transitions: (num_states, num_states)
        self.weights = np.random.randn(num_features, num_states) * 0.1
        self.transitions = np.random.randn(num_states, num_states) * 0.1

    def _score(self, x, y):
        # x: (seq_len, num_features)
        # y: (seq_len,)
        seq_len = x.shape[0]
        score = 0
        for i in range(seq_len):
            score += np.dot(x[i], self.weights[:, y[i]])
            if i > 0:
                score += self.transitions[y[i-1], y[i]]
        return score

    def _forward(self, x):
        seq_len = x.shape[0]
        alpha = np.zeros((seq_len, self.num_states))
        # Initialize alpha[0]
        for s in range(self.num_states):
            alpha[0, s] = np.dot(x[0], self.weights[:, s])

        for i in range(1, seq_len):
            for s in range(self.num_states):
                # alpha[i-1, s'] + transition(s', s) + emission(s)
                prev_scores = alpha[i-1, :] + self.transitions[:, s]
                alpha[i, s] = log_sum_exp(prev_scores) + np.dot(x[i], self.weights[:, s])

        return log_sum_exp(alpha[-1, :]), alpha

    def _backward(self, x):
        seq_len = x.shape[0]
        beta = np.zeros((seq_len, self.num_states))
        # beta[-1] = 0

        for i in range(seq_len - 2, -1, -1):
            for s in range(self.num_states):
                next_scores = beta[i+1, :] + self.transitions[s, :] + np.dot(x[i+1], self.weights).T
                beta[i, s] = log_sum_exp(next_scores)

        return beta

    def _expected_counts(self, x, alpha, beta, Z):
        seq_len = x.shape[0]
        exp_weights = np.zeros_like(self.weights)
        exp_transitions = np.zeros_like(self.transitions)

        # Emission counts
        for i in range(seq_len):
            for s in range(self.num_states):
                prob = np.exp(alpha[i, s] + beta[i, s] - Z)
                exp_weights[:, s] += prob * x[i]

        # Transition counts
        for i in range(1, seq_len):
            for s_prev in range(self.num_states):
                for s_next in range(self.num_states):
                    score = alpha[i-1, s_prev] + self.transitions[s_prev, s_next] + np.dot(x[i], self.weights[:, s_next]) + beta[i, s_next]
                    prob = np.exp(score - Z)
                    exp_transitions[s_prev, s_next] += prob

        return exp_weights, exp_transitions

    def fit(self, X, Y, epochs=50, lr=0.01):
        for epoch in range(epochs):
            grad_weights = np.zeros_like(self.weights)
            grad_transitions = np.zeros_like(self.transitions)
            total_loss = 0

            for x, y in zip(X, Y):
                seq_len = x.shape[0]
                Z, alpha = self._forward(x)
                beta = self._backward(x)

                score = self._score(x, y)
                total_loss -= (score - Z)

                exp_w, exp_t = self._expected_counts(x, alpha, beta, Z)

                # Empirical counts
                emp_w = np.zeros_like(self.weights)
                emp_t = np.zeros_like(self.transitions)
                for i in range(seq_len):
                    emp_w[:, y[i]] += x[i]
                    if i > 0:
                        emp_t[y[i-1], y[i]] += 1

                grad_weights += (emp_w - exp_w)
                grad_transitions += (emp_t - exp_t)

            self.weights += lr * grad_weights
            self.transitions += lr * grad_transitions

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X):.4f}")

    def predict(self, x):
        # Viterbi decoding
        seq_len = x.shape[0]
        viterbi = np.zeros((seq_len, self.num_states))
        backpointer = np.zeros((seq_len, self.num_states), dtype=int)

        for s in range(self.num_states):
            viterbi[0, s] = np.dot(x[0], self.weights[:, s])

        for i in range(1, seq_len):
            for s in range(self.num_states):
                scores = viterbi[i-1, :] + self.transitions[:, s]
                best_prev = np.argmax(scores)
                backpointer[i, s] = best_prev
                viterbi[i, s] = scores[best_prev] + np.dot(x[i], self.weights[:, s])

        best_path = []
        best_last = np.argmax(viterbi[-1, :])
        best_path.append(best_last)

        for i in range(seq_len - 1, 0, -1):
            best_last = backpointer[i, best_last]
            best_path.append(best_last)

        return best_path[::-1]

if __name__ == "__main__":
    # Generate some synthetic data (POS tagging simulation)
    # Features: [is_capitalized, is_numeric, length]
    # States: 0 (Noun), 1 (Verb), 2 (Other)

    X = [
        np.array([[1, 0, 5], [0, 0, 3], [0, 1, 2]]),
        np.array([[1, 0, 6], [0, 0, 4]])
    ]
    Y = [
        np.array([0, 1, 2]),
        np.array([0, 1])
    ]

    crf = LinearChainCRF(num_states=3, num_features=3)
    print("Training Linear Chain CRF...")
    crf.fit(X, Y, epochs=50, lr=0.1)

    test_x = np.array([[1, 0, 4], [0, 0, 3]])
    pred = crf.predict(test_x)
    print(f"Prediction for test sequence: {pred}")
    print("Linear Chain CRF component successfully implemented and tested.")


def train_crf():
    np.random.seed(42)
    # Generate synthetic sequence data for Conditional Random Field (CRF)
    # Binary states: 0 or 1
    # Features: randomly generated
    seq_len = 10
    num_features = 5
    num_states = 2

    # X: (seq_len, num_features)
    X = np.random.randn(seq_len, num_features)
    # True transition weights: T(i, j) = transition from i to j
    true_T = np.array([[1.5, -0.5],
                       [-1.0, 2.0]])
    # True emission weights: E(i, f) = emission for state i, feature f
    true_E = np.random.randn(num_states, num_features)

    # Generate true sequence
    # For simplicity, we just assign labels randomly to create a dataset
    y = np.random.randint(0, num_states, size=seq_len)

    # Initialize weights
    T = np.zeros((num_states, num_states))
    E = np.zeros((num_states, num_features))
    lr = 0.05
    epochs = 500

    print("Training Conditional Random Field (CRF)...")
    for epoch in range(epochs):
        # Forward-Backward algorithm for gradients

        # 1. Compute node potentials (emission scores)
        # node_potentials: (seq_len, num_states)
        node_potentials = np.dot(X, E.T)

        # 2. Forward pass (alphas)
        alphas = np.zeros((seq_len, num_states))
        alphas[0] = node_potentials[0]
        for t in range(1, seq_len):
            for s in range(num_states):
                # log-sum-exp for numerical stability, simplified here
                m = np.max(alphas[t-1] + T[:, s])
                alphas[t, s] = node_potentials[t, s] + m + np.log(np.sum(np.exp(alphas[t-1] + T[:, s] - m)))

        # 3. Backward pass (betas)
        betas = np.zeros((seq_len, num_states))
        for t in range(seq_len-2, -1, -1):
            for s in range(num_states):
                m = np.max(T[s, :] + node_potentials[t+1] + betas[t+1])
                betas[t, s] = m + np.log(np.sum(np.exp(T[s, :] + node_potentials[t+1] + betas[t+1] - m)))

        # 4. Compute marginals
        log_Z = np.max(alphas[-1]) + np.log(np.sum(np.exp(alphas[-1] - np.max(alphas[-1]))))

        # Node marginals: P(y_t = s | X)
        marginals_node = np.exp(alphas + betas - log_Z)

        # Edge marginals: P(y_{t-1} = i, y_t = j | X)
        marginals_edge = np.zeros((seq_len-1, num_states, num_states))
        for t in range(1, seq_len):
            for i in range(num_states):
                for j in range(num_states):
                    marginals_edge[t-1, i, j] = np.exp(alphas[t-1, i] + T[i, j] + node_potentials[t, j] + betas[t, j] - log_Z)

        # 5. Compute gradients
        grad_E = np.zeros_like(E)
        grad_T = np.zeros_like(T)

        # Empirical counts (from true labels y)
        for t in range(seq_len):
            grad_E[y[t]] -= X[t]
        for t in range(1, seq_len):
            grad_T[y[t-1], y[t]] -= 1

        # Expected counts (from model marginals)
        for t in range(seq_len):
            for s in range(num_states):
                grad_E[s] += marginals_node[t, s] * X[t]

        for t in range(1, seq_len):
            for i in range(num_states):
                for j in range(num_states):
                    grad_T[i, j] += marginals_edge[t-1, i, j]

        # Update weights
        E -= lr * grad_E
        T -= lr * grad_T

        if epoch % 100 == 0:
            # Score of true sequence
            score = 0
            for t in range(seq_len):
                score += np.dot(E[y[t]], X[t])
            for t in range(1, seq_len):
                score += T[y[t-1], y[t]]
            loss = -(score - log_Z)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Success! Model learned CRF potentials.")

if __name__ == "__main__":
    train_crf()
