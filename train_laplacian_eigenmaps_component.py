import numpy as np

class LaplacianEigenmaps:
    def __init__(self, n_components=2, n_neighbors=5, gamma=1.0):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.embedding_ = None

    def fit_transform(self, X):
        n_samples = X.shape[0]

        sum_X = np.sum(X**2, axis=1)
        D_sq = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T)
        D_sq = np.maximum(D_sq, 0)

        neighbors = np.argsort(D_sq, axis=1)[:, 1:self.n_neighbors+1]

        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in neighbors[i]:
                W[i, j] = np.exp(-self.gamma * D_sq[i, j])
                W[j, i] = W[i, j]

        D_diag = np.sum(W, axis=1)
        D = np.diag(D_diag)
        L = D - W

        D_inv_sqrt = np.diag(1.0 / np.sqrt(D_diag + 1e-12))
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt

        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvectors = D_inv_sqrt @ eigenvectors

        self.embedding_ = eigenvectors[:, 1:self.n_components+1]
        return self.embedding_

if __name__ == "__main__":
    np.random.seed(42)
    t1 = np.linspace(0, 2*np.pi, 100)
    X1 = np.vstack([np.cos(t1), np.sin(t1)]).T + np.random.randn(100, 2) * 0.05
    t2 = np.linspace(0, 2*np.pi, 100)
    X2 = 0.5 * np.vstack([np.cos(t2), np.sin(t2)]).T + np.random.randn(100, 2) * 0.05
    X = np.vstack([X1, X2])

    print("Original data shape:", X.shape)
    le = LaplacianEigenmaps(n_components=2, n_neighbors=10, gamma=10.0)
    X_embedded = le.fit_transform(X)
    print("Embedded data shape:", X_embedded.shape)
    print("Optimization finished successfully.")
