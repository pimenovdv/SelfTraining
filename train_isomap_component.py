import numpy as np

class Isomap:
    def __init__(self, n_neighbors=5, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def fit_transform(self, X):
        n_samples = X.shape[0]
        # Step 1: k-nearest neighbors
        sq_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
        dists = np.sqrt(sq_dists)

        graph = np.full((n_samples, n_samples), np.inf)
        np.fill_diagonal(graph, 0)
        for i in range(n_samples):
            indices = np.argsort(dists[i])[1:self.n_neighbors+1]
            graph[i, indices] = dists[i, indices]
            graph[indices, i] = dists[i, indices]

        # Step 2: Shortest paths (Floyd-Warshall)
        for k in range(n_samples):
            graph = np.minimum(graph, graph[:, k:k+1] + graph[k:k+1, :])

        # Step 3: Multi-Dimensional Scaling (MDS)
        H = np.eye(n_samples) - (1/n_samples) * np.ones((n_samples, n_samples))
        B = -0.5 * H.dot(graph ** 2).dot(H)

        eigenvalues, eigenvectors = np.linalg.eigh(B)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues = eigenvalues[:self.n_components]
        eigenvectors = eigenvectors[:, :self.n_components]

        eigenvalues = np.maximum(eigenvalues, 0)

        return eigenvectors * np.sqrt(eigenvalues)

if __name__ == "__main__":
    np.random.seed(42)
    t = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, 200)
    y = np.random.uniform(0, 10, 200)
    x = t * np.cos(t)
    z = t * np.sin(t)
    X = np.column_stack((x, y, z))

    print("Dataset shape:", X.shape)

    isomap = Isomap(n_neighbors=10, n_components=2)
    X_reduced = isomap.fit_transform(X)

    print("Reduced shape:", X_reduced.shape)
    print("First 5 points:\n", X_reduced[:5])
    print("Isomap component successfully executed.")
