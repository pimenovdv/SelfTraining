import numpy as np

class FactorAnalysis:
    def __init__(self, n_components, tol=1e-4, max_iter=1000):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        n_samples, n_features = X.shape

        self.components_ = np.random.randn(self.n_components, n_features)
        self.noise_variance_ = np.ones(n_features)

        for i in range(self.max_iter):
            W = self.components_
            Psi = np.diag(self.noise_variance_)

            M = W.T @ W + Psi

            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                M_inv = np.linalg.pinv(M)

            beta = W @ M_inv
            E_z = X_centered @ beta.T
            E_zz = np.eye(self.n_components) - beta @ W.T + (E_z.T @ E_z) / n_samples

            W_new = (X_centered.T @ E_z) @ np.linalg.inv(n_samples * E_zz)
            Psi_new = np.diag(X_centered.T @ X_centered) / n_samples - np.diag(W_new @ E_z.T @ X_centered) / n_samples
            Psi_new = np.maximum(Psi_new, 1e-6)

            if np.allclose(self.components_, W_new.T, atol=self.tol) and \
               np.allclose(self.noise_variance_, Psi_new, atol=self.tol):
                self.components_ = W_new.T
                self.noise_variance_ = Psi_new
                break

            self.components_ = W_new.T
            self.noise_variance_ = Psi_new

    def transform(self, X):
        X_centered = X - self.mean_
        W = self.components_
        Psi = np.diag(self.noise_variance_)
        M = W.T @ W + Psi
        M_inv = np.linalg.inv(M)
        beta = W @ M_inv
        return X_centered @ beta.T

if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 1000
    z = np.random.randn(n_samples, 2)
    W_true = np.random.randn(5, 2)
    noise = np.random.randn(n_samples, 5) * 0.5
    X = z @ W_true.T + noise

    fa = FactorAnalysis(n_components=2)
    fa.fit(X)

    z_pred = fa.transform(X)
    print("Factor Analysis Components:\n", np.round(fa.components_, 3))
    print("\nNoise Variance:\n", np.round(fa.noise_variance_, 3))
