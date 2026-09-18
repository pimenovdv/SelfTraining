import numpy as np

def train_swa(X, y, hidden_size, epochs, lr, swa_start):
    np.random.seed(42)
    input_size = X.shape[1]
    output_size = y.shape[1]

    # Initialize weights
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((1, output_size))

    # SWA weights
    swa_W1 = np.copy(W1)
    swa_b1 = np.copy(b1)
    swa_W2 = np.copy(W2)
    swa_b2 = np.copy(b2)
    swa_n = 0

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_deriv(a):
        return a * (1 - a)

    def forward(X, W1, b1, W2, b2):
        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        return A1, A2

    for epoch in range(epochs):
        # Forward pass
        A1, A2 = forward(X, W1, b1, W2, b2)

        # Loss (MSE)
        loss = np.mean((A2 - y) ** 2)

        # Backward pass
        dA2 = (A2 - y) / y.shape[0]
        dZ2 = dA2 * sigmoid_deriv(A2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * sigmoid_deriv(A1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights (SGD)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # Update SWA weights
        if epoch >= swa_start:
            swa_W1 = (swa_W1 * swa_n + W1) / (swa_n + 1)
            swa_b1 = (swa_b1 * swa_n + b1) / (swa_n + 1)
            swa_W2 = (swa_W2 * swa_n + W2) / (swa_n + 1)
            swa_b2 = (swa_b2 * swa_n + b2) / (swa_n + 1)
            swa_n += 1

        if epoch % 5000 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.6f}")

    # Evaluate regular model
    _, A2 = forward(X, W1, b1, W2, b2)
    regular_loss = np.mean((A2 - y) ** 2)

    # Evaluate SWA model
    _, swa_A2 = forward(X, swa_W1, swa_b1, swa_W2, swa_b2)
    swa_loss = np.mean((swa_A2 - y) ** 2)

    print(f"Final Regular Loss: {regular_loss:.6f}")
    print(f"Final SWA Loss: {swa_loss:.6f}")

    print("SWA Predictions:")
    print(swa_A2)

    return regular_loss, swa_loss

def test_component():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    train_swa(X, y, hidden_size=8, epochs=20000, lr=1.0, swa_start=15000)

if __name__ == "__main__":
    test_component()
