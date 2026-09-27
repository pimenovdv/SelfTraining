import numpy as np
import argparse

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Training loop
def train_qhm_ffn(X, y, hidden_size, epochs, learning_rate, momentum=0.9, nu=0.7):
    input_size = X.shape[1]
    output_size = y.shape[1]

    # Initialize weights and biases randomly with mean 0
    np.random.seed(42) # For reproducibility
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))

    # QHM cache initialization
    g_W1 = np.zeros_like(W1)
    g_b1 = np.zeros_like(b1)
    g_W2 = np.zeros_like(W2)
    g_b2 = np.zeros_like(b2)

    for epoch in range(1, epochs + 1):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Loss calculation (Mean Squared Error)
        loss = np.mean(0.5 * (a2 - y) ** 2)

        if epoch % (epochs // 10) == 0 or epoch == epochs:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

        # Backward pass
        # Error at output
        dZ2 = (a2 - y) * sigmoid_derivative(z2)
        dW2 = np.dot(a1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

        # Error at hidden layer
        dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(z1)
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

        # QHM updates
        g_W1 = momentum * g_W1 + (1 - momentum) * dW1
        step_W1 = (1 - nu) * dW1 + nu * g_W1
        W1 -= learning_rate * step_W1

        g_b1 = momentum * g_b1 + (1 - momentum) * db1
        step_b1 = (1 - nu) * db1 + nu * g_b1
        b1 -= learning_rate * step_b1

        g_W2 = momentum * g_W2 + (1 - momentum) * dW2
        step_W2 = (1 - nu) * dW2 + nu * g_W2
        W2 -= learning_rate * step_W2

        g_b2 = momentum * g_b2 + (1 - momentum) * db2
        step_b2 = (1 - nu) * db2 + nu * g_b2
        b2 -= learning_rate * step_b2

    return W1, b1, W2, b2, a2

def test_qhm_component():
    parser = argparse.ArgumentParser(description="Train a simple FFN using QHM Optimizer.")
    parser.add_argument("--hidden_size", type=int, default=8, help="Number of neurons in hidden layer.")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2.0, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum factor.")
    parser.add_argument("--nu", type=float, default=0.7, help="Nu factor.")
    args = parser.parse_args()

    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print(f"Training FFN with QHM (hidden_size={args.hidden_size}, epochs={args.epochs}, lr={args.lr}, momentum={args.momentum}, nu={args.nu})")

    W1, b1, W2, b2, predictions = train_qhm_ffn(X, y, args.hidden_size, args.epochs, args.lr, args.momentum, args.nu)

    print("\nTraining Complete.")
    print("Final Predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")

if __name__ == "__main__":
    test_qhm_component()
