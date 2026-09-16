import numpy as np
import os
import argparse

# GEGLU approximation activation and its derivative
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def gelu_derivative(x):
    u = np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))
    du = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * np.power(x, 2))
    y = np.tanh(u)
    dy = (1 - y**2) * du
    return 0.5 * (1 + y) + 0.5 * x * dy

# Sigmoid activation and its derivative for the output layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Training loop
def train_geglu(X, y, hidden_size, epochs, learning_rate):
    input_size = X.shape[1]
    output_size = y.shape[1]

    # Initialize weights and biases randomly with mean 0
    np.random.seed(42) # For reproducibility
    W = np.random.randn(input_size, hidden_size) * 0.1
    b = np.zeros((1, hidden_size))
    V = np.random.randn(input_size, hidden_size) * 0.1
    c = np.zeros((1, hidden_size))

    U = np.random.randn(hidden_size, output_size) * 0.1
    d = np.zeros((1, output_size))

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W) + b
        z2 = np.dot(X, V) + c

        # GEGLU activation
        a1 = gelu(z1)
        h = a1 * z2

        # Output layer
        z3 = np.dot(h, U) + d
        a3 = sigmoid(z3)

        # Loss calculation (Mean Squared Error)
        loss = np.mean(0.5 * (a3 - y) ** 2)

        if (epoch) % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

        # Backward pass
        # Error at output
        dZ3 = (a3 - y) * sigmoid_derivative(z3)
        dU = np.dot(h.T, dZ3) / X.shape[0]
        dd = np.sum(dZ3, axis=0, keepdims=True) / X.shape[0]

        # Error at GEGLU layer
        dh = np.dot(dZ3, U.T)

        # GEGLU splits into z1 and z2
        dZ2 = dh * a1
        dZ1 = dh * z2 * gelu_derivative(z1)

        # Gradients for W, b, V, c
        dW = np.dot(X.T, dZ1) / X.shape[0]
        db = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
        dV = np.dot(X.T, dZ2) / X.shape[0]
        dc = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

        # Update weights and biases
        W -= learning_rate * dW
        b -= learning_rate * db
        V -= learning_rate * dV
        c -= learning_rate * dc
        U -= learning_rate * dU
        d -= learning_rate * dd

    return W, b, V, c, U, d, a3

def main():
    parser = argparse.ArgumentParser(description="Train a GEGLU Network on XOR dataset.")
    parser.add_argument("--hidden_size", type=int, default=8, help="Number of neurons in hidden layer.")
    parser.add_argument("--epochs", type=int, default=50000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate.")
    args = parser.parse_args()

    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print(f"Training GEGLU Component with hidden_size={args.hidden_size}, epochs={args.epochs}, lr={args.lr}")

    W, b, V, c, U, d, predictions = train_geglu(X, y, args.hidden_size, args.epochs, args.lr)

    print("\nTraining Complete.")
    print("Final Predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")

if __name__ == "__main__":
    main()
