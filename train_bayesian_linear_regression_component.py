"""
Bayesian Linear Regression
This script implements Bayesian Linear Regression to estimate uncertainty in predictions.
"""

import numpy as np

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        # alpha: precision of the prior (lambda = alpha * I)
        # beta: precision of the noise (beta = 1 / sigma^2)
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_cov = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Prior covariance matrix inverse: alpha * I
        S_0_inv = self.alpha * np.eye(n_features)

        # Posterior covariance matrix inverse: S_N_inv = S_0_inv + beta * X.T @ X
        S_N_inv = S_0_inv + self.beta * X.T @ X

        # Posterior covariance matrix
        self.w_cov = np.linalg.inv(S_N_inv)

        # Posterior mean: m_N = beta * S_N @ X.T @ y
        self.w_mean = self.beta * self.w_cov @ X.T @ y

    def predict(self, X, return_std=False):
        y_pred = X @ self.w_mean

        if return_std:
            # Predictive variance: sigma^2_N(x) = 1/beta + x.T @ S_N @ x
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y_pred, y_std

        return y_pred

if __name__ == "__main__":
    np.random.seed(42)
    print("Testing Bayesian Linear Regression...")

    # Generate linear data with noise
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    # Add bias term
    X_b = np.c_[np.ones((100, 1)), X]

    # True weights: [2, 3]
    true_w = np.array([2.0, 3.0])
    noise_std = 0.5
    y = X_b @ true_w + np.random.randn(100) * noise_std

    # Train Bayesian Linear Regression
    # beta = 1 / noise_std^2
    model = BayesianLinearRegression(alpha=1.0, beta=1/(noise_std**2))
    model.fit(X_b, y)

    # Predict with uncertainty
    y_pred, y_std = model.predict(X_b, return_std=True)

    mse = np.mean((y - y_pred)**2)

    print(f"MSE: {mse:.4f}")
    print(f"Learned Mean Weights: Bias={model.w_mean[0]:.4f} (True: 2.0), W={model.w_mean[1]:.4f} (True: 3.0)")
    print(f"Average Predictive Std: {np.mean(y_std):.4f} (True Noise Std: {noise_std})")

    if mse < 1.0 and abs(np.mean(y_std) - noise_std) < 0.2:
        print("Bayesian Linear Regression implementation is successful.")
    else:
        print("Bayesian Linear Regression implementation failed.")
