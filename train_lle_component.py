import numpy as np

class LocallyLinearEmbedding:
    def __init__(self, n_neighbors=5, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def fit_transform(self, X):
        n_samples = X.shape[0]

        # Step 1: k-nearest neighbors
        sq_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
        neighbors = np.argsort(sq_dists, axis=1)[:, 1:self.n_neighbors+1]

        # Step 2: Reconstruction weights
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            Z = X[neighbors[i]] - X[i]
            C = Z.dot(Z.T)
            C += np.eye(self.n_neighbors) * 1e-3 * np.trace(C)
            w = np.linalg.solve(C, np.ones(self.n_neighbors))
            w = w / np.sum(w)
            W[i, neighbors[i]] = w

        # Step 3: Compute embedding
        I = np.eye(n_samples)
        M = (I - W).T.dot(I - W)

        eigenvalues, eigenvectors = np.linalg.eigh(M)

        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Skip the first eigenvector (which is all ones, corresponding to eigenvalue 0)
        return eigenvectors[:, 1:self.n_components+1]

if __name__ == "__main__":
    np.random.seed(42)
    t = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, 200)
    y = np.random.uniform(0, 10, 200)
    x = t * np.cos(t)
    z = t * np.sin(t)
    X = np.column_stack((x, y, z))

    print("Dataset shape:", X.shape)

    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
    X_reduced = lle.fit_transform(X)

    print("Reduced shape:", X_reduced.shape)
    print("First 5 points:\n", X_reduced[:5])
    print("LLE component successfully executed.")
