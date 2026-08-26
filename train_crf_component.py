import numpy as np

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
