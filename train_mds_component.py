import numpy as np
import os

class Module:
    pass

class MDSComponent(Module):
    """
    Classical Multidimensional Scaling (MDS).
    Finds a low-dimensional representation of data that preserves pairwise Euclidean distances
    using eigendecomposition of the doubly-centered distance matrix.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]

        # 1. Compute squared pairwise Euclidean distances
        D_sq = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                D_sq[i, j] = np.sum((X[i] - X[j])**2)

        # 2. Double centering
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        B = -0.5 * H.dot(D_sq).dot(H)

        # 3. Eigendecomposition of B
        eigenvalues, eigenvectors = np.linalg.eigh(B)

        # 4. Sort eigenvalues and descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Take the top n_components
        top_eigenvalues = eigenvalues[:self.n_components]
        top_eigenvectors = eigenvectors[:, :self.n_components]

        # Ensure eigenvalues are non-negative
        top_eigenvalues = np.maximum(top_eigenvalues, 0)

        # 6. Compute low-dimensional embeddings
        Y = top_eigenvectors.dot(np.diag(np.sqrt(top_eigenvalues)))
        return Y

if __name__ == "__main__":
    print("Testing Classical Multidimensional Scaling (MDS) Component...")
    np.random.seed(42)

    X = np.random.rand(10, 3)
    mds = MDSComponent(n_components=2)
    Y = mds.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {Y.shape}")

    orig_dist = np.linalg.norm(X[0] - X[1])
    new_dist = np.linalg.norm(Y[0] - Y[1])
    print(f"Original distance between pt 0 and 1: {orig_dist:.4f}")
    print(f"Reduced distance between pt 0 and 1: {new_dist:.4f}")

    print("MDS Component verification complete.")
