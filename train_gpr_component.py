import numpy as np

class GaussianProcessRegression:
    def __init__(self, kernel, noise_var=1e-5):
        self.kernel = kernel
        self.noise_var = noise_var
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self.kernel(X, X) + self.noise_var * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test):
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)

        mu = K_s.T.dot(self.K_inv).dot(self.y_train)
        cov = K_ss - K_s.T.dot(self.K_inv).dot(K_s)
        return mu, cov

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return variance * np.exp(-0.5 / length_scale**2 * sqdist)

def test_gpr():
    print("Testing Gaussian Process Regression Component...")
    np.random.seed(42)
    X_train = np.random.uniform(-5, 5, (10, 1))
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, (10, 1))

    gpr = GaussianProcessRegression(kernel=rbf_kernel, noise_var=0.1)
    gpr.fit(X_train, y_train)

    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
    mu, cov = gpr.predict(X_test)

    assert mu.shape == (100, 1), "Mean shape mismatch."
    assert cov.shape == (100, 100), "Covariance shape mismatch."

    # Test accuracy on train data
    mu_train, _ = gpr.predict(X_train)
    mse = np.mean((mu_train - y_train)**2)
    print(f"Train MSE: {mse:.4f}")
    assert mse < 0.5, f"MSE {mse} is too high."

    print("GPR testing successful!")

if __name__ == "__main__":
    test_gpr()
