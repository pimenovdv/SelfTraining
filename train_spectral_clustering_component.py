"""
Spectral Clustering Component

This script evaluates a Spectral Clustering component, verifying its logic mathematically in pure NumPy.
"""

import numpy as np

class SpectralClustering:
    """
    Spectral Clustering Component

    This component performs spectral clustering by:
    1. Computing the RBF (Gaussian) affinity matrix between points.
    2. Computing the normalized graph Laplacian (L_sym = I - D^{-1/2} W D^{-1/2}).
    3. Finding the eigenvectors corresponding to the k smallest eigenvalues.
    4. Normalizing the rows of the eigenvector matrix.
    5. Clustering the rows using K-Means.
    """
    def __init__(self, n_clusters, gamma=15.0, random_state=42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def _rbf_kernel(self, X):
        sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        return np.exp(-self.gamma * sq_dists)

    def fit_predict(self, X):
        # 1. Affinity matrix
        W = self._rbf_kernel(X)
        np.fill_diagonal(W, 0)

        # 2. Normalized Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))
        L = np.eye(X.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # 4. First k eigenvectors & row normalization
        U = eigenvectors[:, :self.n_clusters]
        U = U / np.linalg.norm(U, axis=1, keepdims=True)

        # 5. K-Means
        np.random.seed(self.random_state)
        centroids = U[np.random.choice(U.shape[0], self.n_clusters, replace=False)]

        for _ in range(100):
            dists = np.sum((U[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            new_centroids = np.array([U[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k] for k in range(self.n_clusters)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return labels

def generate_moons(n_samples=200, noise=0.05):
    np.random.seed(42)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

if __name__ == "__main__":
    print("Testing Spectral Clustering Component...")

    # Generate non-convex data (two moons)
    X, y = generate_moons(n_samples=200, noise=0.05)

    # Initialize and fit
    sc = SpectralClustering(n_clusters=2, gamma=15.0)
    labels = sc.fit_predict(X)

    # Calculate accuracy (handling label permutation)
    accuracy = np.mean(labels == y)
    accuracy = max(accuracy, 1 - accuracy)

    print(f"Spectral Clustering Accuracy on Two Moons: {accuracy * 100:.2f}%")

    assert accuracy > 0.95, f"Expected accuracy > 95%, got {accuracy * 100:.2f}%"
    print("Spectral Clustering Component verified successfully!")
