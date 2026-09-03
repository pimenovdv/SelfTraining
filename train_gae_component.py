import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

def train_gae():
    print("Training Graph Autoencoder (GAE) component...")

    # Synthetic graph (e.g., 4 nodes)
    # Adjacency matrix A (undirected, unweighted) with self-loops
    A = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ])

    # Node features X (4 nodes, 3 features)
    X = np.array([
        [1.0, 0.2, 0.3],
        [0.8, 0.4, 0.1],
        [0.2, 0.9, 0.8],
        [0.1, 0.8, 0.9]
    ])

    num_nodes = X.shape[0]
    input_dim = X.shape[1]
    hidden_dim = 2

    # Degree matrix and symmetrically normalized adjacency matrix
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    np.random.seed(42)
    # GCN Encoder weights
    W = np.random.randn(input_dim, hidden_dim) * 0.1

    lr = 0.5
    epochs = 1000

    for epoch in range(epochs):
        # Forward pass (Encoder)
        Z = A_norm @ X @ W

        # Forward pass (Decoder)
        A_hat_logits = Z @ Z.T
        A_hat = sigmoid(A_hat_logits)

        # Binary Cross Entropy Loss
        loss = -np.mean(A * np.log(A_hat + 1e-9) + (1 - A) * np.log(1 - A_hat + 1e-9))

        # Backward pass
        dLogits = (A_hat - A) / (num_nodes * num_nodes)

        # dLogits = d(Z @ Z.T) -> dZ = dLogits @ Z + dLogits.T @ Z
        dZ = (dLogits + dLogits.T) @ Z

        # dZ = A_norm @ X @ dW -> dW = X.T @ A_norm.T @ dZ
        dW = X.T @ A_norm.T @ dZ

        W -= lr * dW

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Final Reconstructed Adjacency Matrix:")
    print(np.round(A_hat, 2))
    print("Target Adjacency Matrix:")
    print(A)

    mse = np.mean((A_hat - A)**2)
    print(f"MSE: {mse}")
    assert mse < 0.1, "GAE failed to reconstruct the adjacency matrix well."
    print("Graph Autoencoder (GAE) component trained and evaluated successfully.")

if __name__ == "__main__":
    train_gae()
