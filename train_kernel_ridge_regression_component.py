import numpy as np

def rbf_kernel(X1, X2, gamma=0.1):
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

class KernelRidgeRegression:
    def __init__(self, alpha=1.0, gamma=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.X_train = None
        self.dual_coef = None

    def fit(self, X, y):
        self.X_train = X
        K = rbf_kernel(X, X, self.gamma)
        n_samples = X.shape[0]
        # (K + alpha * I) * dual_coef = y
        self.dual_coef = np.linalg.solve(K + self.alpha * np.eye(n_samples), y)

    def predict(self, X):
        K_trans = rbf_kernel(X, self.X_train, self.gamma)
        return np.dot(K_trans, self.dual_coef)

if __name__ == "__main__":
    print("Testing Kernel Ridge Regression Component...")

    # Generate some non-linear synthetic data
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(20)) # Add noise

    # Train
    model = KernelRidgeRegression(alpha=0.1, gamma=0.5)
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)

    # Calculate Mean Squared Error
    mse = np.mean((y - y_pred)**2)
    print(f"Mean Squared Error: {mse:.4f}")

    if mse < 0.5:
        print("Kernel Ridge Regression successfully learned the non-linear relationship.")
    else:
        print("Kernel Ridge Regression failed to learn the relationship.")
        exit(1)
