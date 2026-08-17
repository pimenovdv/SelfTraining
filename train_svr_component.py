"""
Support Vector Regression (SVR)
This script implements Support Vector Regression using a linear kernel and subgradient descent.
"""

import numpy as np

class SupportVectorRegression:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, epsilon=0.1, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition_high = y[idx] - np.dot(x_i, self.w) - self.b > self.epsilon
                condition_low = y[idx] - np.dot(x_i, self.w) - self.b < -self.epsilon

                if condition_high:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - x_i)
                    self.b -= self.lr * (-1)
                elif condition_low:
                    self.w -= self.lr * (2 * self.lambda_param * self.w + x_i)
                    self.b -= self.lr * (1)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)

    def predict(self, X):
        return np.dot(X, self.w) + self.b

if __name__ == "__main__":
    np.random.seed(42)
    print("Testing Support Vector Regression...")

    # Generate linear data with noise
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = (3 * X + 2 + np.random.randn(100, 1) * 0.5).ravel()

    # Train SVR
    model = SupportVectorRegression(learning_rate=0.01, lambda_param=0.01, epsilon=0.2, n_iters=1000)
    model.fit(X, y)

    preds = model.predict(X)
    mse = np.mean((y - preds)**2)

    print(f"MSE: {mse:.4f}")

    # Verify the learned parameters
    print(f"Learned Weight: {model.w[0]:.4f} (True: 3.0)")
    print(f"Learned Bias: {model.b:.4f} (True: 2.0)")

    if mse < 1.0:
        print("SVR implementation is successful.")
    else:
        print("SVR implementation failed to converge properly.")
