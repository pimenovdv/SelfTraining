import numpy as np
import os
import argparse

np.random.seed(42)

class RBFNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.centroids = np.random.randn(hidden_dim, input_dim)
        self.sigma = np.ones((hidden_dim,)) * 1.0

        self.W_out = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b_out = np.zeros((1, output_dim))

    def forward(self, X):
        diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        self.dist_sq = np.sum(diff**2, axis=2)

        self.H = np.exp(-self.dist_sq / (2 * self.sigma**2))
        out = np.dot(self.H, self.W_out) + self.b_out
        return out

    def backward(self, X, out, Y, lr):
        m = X.shape[0]

        d_out = 2 * (out - Y) / m

        dW_out = np.dot(self.H.T, d_out)
        db_out = np.sum(d_out, axis=0, keepdims=True)

        dH = np.dot(d_out, self.W_out.T)

        d_dist_sq = dH * self.H * (-1 / (2 * self.sigma**2))

        diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        dcentroids = np.sum(d_dist_sq[:, :, np.newaxis] * (-2 * diff), axis=0)

        dsigma = np.sum(dH * self.H * (self.dist_sq / (self.sigma**3)), axis=0)

        self.W_out -= lr * dW_out
        self.b_out -= lr * db_out
        self.centroids -= lr * dcentroids

        self.sigma = np.maximum(self.sigma - lr * dsigma, 1e-4)

def train_test():
    parser = argparse.ArgumentParser(description="Train a Radial Basis Function (RBF) Network.")
    args = parser.parse_args()

    X = np.random.uniform(-1, 1, (200, 2))
    Y = np.zeros((200, 1))
    for i in range(200):
        if X[i, 0] * X[i, 1] > 0:
            Y[i, 0] = 1.0
        else:
            Y[i, 0] = 0.0

    print("Training Radial Basis Function (RBF) Network...")
    model = RBFNetwork(2, 20, 1)

    for epoch in range(2000):
        out = model.forward(X)
        model.backward(X, out, Y, 0.1)

    loss = np.mean((out - Y)**2)
    print(f"Final Loss: {loss:.6f}")

    if loss < 0.1:
        print("Success! Model learned non-linear boundaries using RBFs.")

        docs_dir = "docs"
        os.makedirs(docs_dir, exist_ok=True)
        report_path = os.path.join(docs_dir, "0072_train_rbf_component.md")

        report_content = f"""# 0072_train_rbf_component

## Status
Success

## Component
Radial Basis Function (RBF) Network

## Description
Implemented and evaluated a Radial Basis Function (RBF) Network component using pure NumPy. This component tests learning non-linear boundaries using localized Gaussian basis functions, validating the optimization of centroids, widths, and output weights via manual backpropagation.

## Results
- **Final Loss (MSE):** {loss:.6f}

The model successfully learned the non-linear dataset boundaries.

**Script:** `train_rbf_component.py`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\nExperiment report saved to {report_path}")
    else:
        print("Failed.")

if __name__ == "__main__":
    train_test()
