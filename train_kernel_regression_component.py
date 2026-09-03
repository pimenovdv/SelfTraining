import numpy as np

def rbf_kernel(x1, x2, gamma=1.0):
    sq_dist = np.sum(x1**2, axis=1, keepdims=True) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
    return np.exp(-gamma * sq_dist)

class KernelRegression:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        K = rbf_kernel(X, self.X_train, self.gamma)
        weights = K / (np.sum(K, axis=1, keepdims=True) + 1e-8)
        return np.dot(weights, self.y_train)

if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.sort(np.random.rand(100, 1) * 10, axis=0)
    y_train = np.sin(X_train).ravel() + np.random.randn(100) * 0.1

    kr = KernelRegression(gamma=5.0)
    kr.fit(X_train, y_train)

    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = kr.predict(X_test)
    y_true = np.sin(X_test).ravel()

    mse = np.mean((y_pred - y_true)**2)
    print(f"Kernel Regression MSE: {mse:.4f}")
    assert mse < 0.1, f"MSE {mse} is too high, model failed to learn the non-linear relationship."
    print("Kernel Regression component successfully trained and verified.")
