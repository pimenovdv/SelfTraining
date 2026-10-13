import numpy as np
import os
import json
import argparse
from scipy.special import erf

# SERF (SElu-eRF) activation and its derivative
# f(x) = x * erf(ln(1 + e^x))
# We use softplus: sp(x) = ln(1 + e^x) to avoid overflow
# f'(x) = erf(sp(x)) + x * (2 / sqrt(pi)) * e^(-sp(x)^2) * sigmoid(x)

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def serf(x):
    return x * erf(softplus(x))

def serf_derivative(x):
    sp = softplus(x)
    return erf(sp) + x * (2 / np.sqrt(np.pi)) * np.exp(-(sp**2)) * sigmoid(x)

# Training loop
def train_serf_ffn(X, y, hidden_size, epochs, learning_rate):
    input_size = X.shape[1]
    output_size = y.shape[1]

    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = serf(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Loss (MSE)
        loss = np.mean(np.square(a2 - y))

        # Backward pass
        dz2 = (a2 - y) * a2 * (1 - a2)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * serf_derivative(z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    return loss, W1, b1, W2, b2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SERF Component")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    # Simple XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    final_loss, _, _, _, _ = train_serf_ffn(X, y, hidden_size=4, epochs=args.epochs, learning_rate=args.lr)

    print(f"Final Loss: {final_loss:.6f}")

    os.makedirs('results', exist_ok=True)
    with open('results/serf_results.json', 'w') as f:
        json.dump({'loss': float(final_loss)}, f)
