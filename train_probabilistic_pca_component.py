import numpy as np

class ProbabilisticPCA:
    def __init__(self, n_components, tol=1e-4, max_iter=1000):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        n_samples, n_features = X.shape

        self.components_ = np.random.randn(n_features, self.n_components)
        self.noise_variance_ = np.var(X_centered)

        for i in range(self.max_iter):
            W = self.components_
            sigma2 = self.noise_variance_

            M = W.T @ W + sigma2 * np.eye(self.n_components)
            M_inv = np.linalg.inv(M)

            E_z = X_centered @ W @ M_inv
            E_zz = sigma2 * M_inv + (E_z.T @ E_z) / n_samples

            W_new = (X_centered.T @ E_z) @ np.linalg.inv(n_samples * E_zz)

            sigma2_new = np.trace(X_centered.T @ X_centered) / n_samples
            sigma2_new -= 2 * np.trace(E_z.T @ X_centered @ W_new) / n_samples
            sigma2_new += np.trace(E_zz @ (W_new.T @ W_new))
            sigma2_new /= n_features

            sigma2_new = max(sigma2_new, 1e-6)

            if np.allclose(self.components_, W_new, atol=self.tol) and \
               np.abs(self.noise_variance_ - sigma2_new) < self.tol:
                self.components_ = W_new
                self.noise_variance_ = sigma2_new
                break

            self.components_ = W_new
            self.noise_variance_ = sigma2_new

    def transform(self, X):
        X_centered = X - self.mean_
        W = self.components_
        M = W.T @ W + self.noise_variance_ * np.eye(self.n_components)
        M_inv = np.linalg.inv(M)
        return X_centered @ W @ M_inv

if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 1000
    z = np.random.randn(n_samples, 2)
    W_true = np.random.randn(5, 2)
    noise = np.random.randn(n_samples, 5) * 0.5
    X = z @ W_true.T + noise

    ppca = ProbabilisticPCA(n_components=2)
    ppca.fit(X)

    z_pred = ppca.transform(X)
    print("PPCA Components:\n", np.round(ppca.components_, 3))
    print("\nPPCA Noise Variance:\n", np.round(ppca.noise_variance_, 3))
