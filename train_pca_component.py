import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

def test_pca():
    print("Testing PCA Component...")
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[:, 2] = 2 * X[:, 0] + 0.5 * X[:, 1]

    pca = PCA(n_components=2)
    pca.fit(X)
    X_transformed = pca.transform(X)

    assert X_transformed.shape == (100, 2), "Transformed shape mismatch."
    assert pca.components.shape == (3, 2), "Components shape mismatch."
    print("PCA testing successful!")

if __name__ == "__main__":
    test_pca()
