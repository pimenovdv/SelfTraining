import numpy as np

class KMedoids:
    def __init__(self, n_clusters=2, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Initialize medoids randomly from data points
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids = X[medoid_indices]

        # Precompute distance matrix
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

        for _ in range(self.max_iter):
            # Assign each point to the closest medoid
            distances_to_medoids = dist_matrix[:, medoid_indices]
            self.labels = np.argmin(distances_to_medoids, axis=1)

            new_medoid_indices = np.copy(medoid_indices)

            for k in range(self.n_clusters):
                cluster_indices = np.where(self.labels == k)[0]
                if len(cluster_indices) == 0:
                    continue

                # Find point in cluster that minimizes sum of distances to other points in cluster
                cluster_dist_matrix = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
                costs = np.sum(cluster_dist_matrix, axis=1)
                best_medoid_idx_in_cluster = np.argmin(costs)
                new_medoid_indices[k] = cluster_indices[best_medoid_idx_in_cluster]

            if np.array_equal(medoid_indices, new_medoid_indices):
                break

            medoid_indices = new_medoid_indices
            self.medoids = X[medoid_indices]

    def predict(self, X_new):
        dist_matrix = np.linalg.norm(X_new[:, np.newaxis] - self.medoids, axis=2)
        return np.argmin(dist_matrix, axis=1)

if __name__ == "__main__":
    X = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
    ])

    print("Training K-Medoids Component...")
    kmedoids = KMedoids(n_clusters=2)
    kmedoids.fit(X)

    print(f"Medoids:\n{kmedoids.medoids}")
    print(f"Labels:\n{kmedoids.labels}")

    predictions = kmedoids.predict(X)
    print(f"Predictions:\n{predictions}")

    assert kmedoids.labels is not None, "Failed to fit K-Medoids model."
    assert kmedoids.medoids.shape == (2, 2), "Failed to find correct number of medoids."
    print("K-Medoids clustering component successfully tested.")
