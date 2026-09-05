import numpy as np

def test_maxout():
    print("Testing Maxout Network...")
    np.random.seed(42)
    # Synthetic XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_dim = 2
    hidden_dim = 4
    num_pieces = 2 # maxout pieces
    output_dim = 1

    # Initialize weights
    W1 = np.random.randn(input_dim, hidden_dim * num_pieces) * 0.1
    b1 = np.zeros((1, hidden_dim * num_pieces))

    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))

    lr = 0.5
    epochs = 10000

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1 # (4, 8)
        # Reshape to (batch_size, hidden_dim, num_pieces)
        z1_reshaped = z1.reshape(-1, hidden_dim, num_pieces)
        # Maxout activation
        a1 = np.max(z1_reshaped, axis=2) # (4, 4)

        # Output layer
        z2 = np.dot(a1, W2) + b2 # (4, 1)
        # Sigmoid
        a2 = 1 / (1 + np.exp(-z2))

        # Loss (BCE)
        loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))

        # Backward pass
        da2 = (a2 - y) / X.shape[0]
        dz2 = da2 # Sigmoid + BCE derivative

        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T) # (4, 4)

        # Maxout derivative: 1 for the max piece, 0 otherwise
        dz1_reshaped = np.zeros_like(z1_reshaped)
        max_indices = np.argmax(z1_reshaped, axis=2) # (4, 4)

        for i in range(X.shape[0]):
            for j in range(hidden_dim):
                dz1_reshaped[i, j, max_indices[i, j]] = da1[i, j]

        dz1 = dz1_reshaped.reshape(-1, hidden_dim * num_pieces)

        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    print(f"Final Loss: {loss:.4f}")
    assert loss < 0.1, "Maxout Network failed to converge on XOR."
    print("Maxout Network test passed.")

if __name__ == "__main__":
    test_maxout()
