import numpy as np
import os
import argparse

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_adamax_ffn(X, y, hidden_size, epochs, learning_rate, beta1=0.9, beta2=0.999):
    input_size = X.shape[1]
    output_size = y.shape[1]

    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))

    m_W1, u_W1 = np.zeros_like(W1), np.zeros_like(W1)
    m_b1, u_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_W2, u_W2 = np.zeros_like(W2), np.zeros_like(W2)
    m_b2, u_b2 = np.zeros_like(b2), np.zeros_like(b2)

    for epoch in range(1, epochs + 1):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        loss = np.mean(0.5 * (a2 - y) ** 2)

        if epoch % (epochs // 10) == 0 or epoch == epochs:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

        # Backward pass
        dZ2 = (a2 - y) * sigmoid_derivative(z2)
        dW2 = np.dot(a1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

        dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(z1)
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

        t = epoch

        # Update W1
        m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
        u_W1 = np.maximum(beta2 * u_W1, np.abs(dW1))
        W1 -= (learning_rate / (1 - beta1 ** t)) * (m_W1 / (u_W1 + 1e-8))

        # Update b1
        m_b1 = beta1 * m_b1 + (1 - beta1) * db1
        u_b1 = np.maximum(beta2 * u_b1, np.abs(db1))
        b1 -= (learning_rate / (1 - beta1 ** t)) * (m_b1 / (u_b1 + 1e-8))

        # Update W2
        m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
        u_W2 = np.maximum(beta2 * u_W2, np.abs(dW2))
        W2 -= (learning_rate / (1 - beta1 ** t)) * (m_W2 / (u_W2 + 1e-8))

        # Update b2
        m_b2 = beta1 * m_b2 + (1 - beta1) * db2
        u_b2 = np.maximum(beta2 * u_b2, np.abs(db2))
        b2 -= (learning_rate / (1 - beta1 ** t)) * (m_b2 / (u_b2 + 1e-8))

    return W1, b1, W2, b2, a2

def main():
    parser = argparse.ArgumentParser(description="Train a simple FFN using Adamax Optimizer.")
    parser.add_argument("--hidden_size", type=int, default=8, help="Number of neurons in hidden layer.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    args = parser.parse_args()

    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print(f"Training FFN with Adamax (hidden_size={args.hidden_size}, epochs={args.epochs}, lr={args.lr})")

    W1, b1, W2, b2, predictions = train_adamax_ffn(X, y, args.hidden_size, args.epochs, args.lr)

    print("\nTraining Complete.")
    print("Final Predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")

if __name__ == "__main__":
    main()
