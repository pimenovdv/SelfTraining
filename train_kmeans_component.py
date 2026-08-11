import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia = None

    def fit(self, X):
        # Randomly initialize centroids by picking points from X
        np.random.seed(42)
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[idx]

        for i in range(self.max_iters):
            # Assign labels based on closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([X[new_labels == j].mean(axis=0) if np.sum(new_labels == j) > 0 else self.centroids[j] for j in range(self.n_clusters)])

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                self.labels = new_labels
                break

            self.centroids = new_centroids
            self.labels = new_labels

        # Calculate inertia (sum of squared distances to closest centroid)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        self.inertia = np.sum(np.min(distances, axis=1)**2)

def generate_data():
    np.random.seed(42)
    X1 = np.random.randn(50, 2) + np.array([0, 0])
    X2 = np.random.randn(50, 2) + np.array([5, 5])
    X3 = np.random.randn(50, 2) + np.array([0, 5])
    return np.vstack([X1, X2, X3])

def test_kmeans():
    print("Testing K-Means Component...")
    X = generate_data()
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    print(f"Final Inertia: {kmeans.inertia:.4f}")
    assert kmeans.inertia < 300, "Inertia is too high, clustering likely failed."
    assert kmeans.centroids.shape == (3, 2), "Centroids shape mismatch."
    assert kmeans.labels.shape == (150,), "Labels shape mismatch."
    print("K-Means clustering successful!")

if __name__ == "__main__":
    test_kmeans()
