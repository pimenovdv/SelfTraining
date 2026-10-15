"""
Stochastic Average Gradient (SAG) Optimizer Component

This script tests the Stochastic Average Gradient (SAG) optimizer mathematically.
SAG achieves a linear convergence rate for strongly convex finite-sum problems
by keeping a memory of the most recently computed gradient for each training example
and updating the parameter using the average of these gradients.
"""

import numpy as np

def generate_data(num_samples=100, num_features=5, seed=42):
    np.random.seed(seed)
    X = np.random.randn(num_samples, num_features)
    true_w = np.random.randn(num_features)
    # Binary labels {-1, 1}
    y = np.sign(X.dot(true_w))
    y[y == 0] = 1
    return X, y, true_w

def compute_loss(X, y, w):
    # Logistic loss
    margins = y * X.dot(w)
    return np.mean(np.log(1 + np.exp(-margins)))

def compute_gradient(X_i, y_i, w):
    # Gradient of logistic loss for a single sample
    margin = y_i * np.dot(X_i, w)
    # Avoid overflow
    if margin > 100:
        return np.zeros_like(w)
    elif margin < -100:
        return -y_i * X_i
    else:
        return -y_i * X_i / (1 + np.exp(margin))

class SAGOtimizer:
    def __init__(self, w_shape, num_samples, lr=0.1):
        self.w = np.zeros(w_shape)
        self.lr = lr
        self.num_samples = num_samples
        self.gradients_memory = np.zeros((num_samples, w_shape[0]))
        self.average_gradient = np.zeros(w_shape)

    def step(self, idx, X_i, y_i):
        new_grad = compute_gradient(X_i, y_i, self.w)
        old_grad = self.gradients_memory[idx]

        self.average_gradient += (new_grad - old_grad) / self.num_samples
        self.gradients_memory[idx] = new_grad

        self.w -= self.lr * self.average_gradient

def main():
    print("Testing SAG Optimizer Component...")

    num_samples = 100
    num_features = 5
    epochs = 100
    lr = 0.1

    X, y, true_w = generate_data(num_samples, num_features)

    optimizer = SAGOtimizer((num_features,), num_samples, lr=lr)

    initial_loss = compute_loss(X, y, optimizer.w)
    print(f"Initial Loss: {initial_loss:.4f}")

    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        for idx in indices:
            optimizer.step(idx, X[idx], y[idx])

    final_loss = compute_loss(X, y, optimizer.w)
    print(f"Final Loss: {final_loss:.4f}")

    assert final_loss < initial_loss, "Loss did not decrease."
    assert final_loss < 0.2, "Loss is still too high."

    print("SAG Optimizer test passed!")

if __name__ == "__main__":
    main()
