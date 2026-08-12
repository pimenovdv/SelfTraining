import numpy as np

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Identifies core points, expands clusters from them, and marks points in sparse regions as noise.
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        n_samples = X.shape[0]
        # -2: unassigned, -1: noise, >= 0: cluster label
        self.labels_ = np.full(n_samples, -2)
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -2:
                continue

            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1 # Mark as noise
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

        return self.labels_

    def _region_query(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels_[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if self.labels_[neighbor_idx] == -1:
                # Was noise, now borders a cluster, so assign to cluster but do not expand
                self.labels_[neighbor_idx] = cluster_id
            elif self.labels_[neighbor_idx] == -2:
                # Unassigned, assign to cluster and expand
                self.labels_[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)

            i += 1

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

    X += np.random.normal(scale=noise, size=X.shape)

    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

    return X, y

if __name__ == "__main__":
    print("Generating non-convex data (two moons)...")
    X, y = generate_moons(n_samples=300, noise=0.1)

    print("Training DBSCAN model...")
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    labels = dbscan.fit(X)

    unique_clusters = set(labels)
    n_clusters = len(unique_clusters) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise}")

    if n_clusters == 2:
        print("Success: DBSCAN correctly identified 2 distinct non-convex clusters.")
    else:
        print(f"Warning: Expected 2 clusters, found {n_clusters}.")
