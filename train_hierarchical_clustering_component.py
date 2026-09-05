import numpy as np

def compute_distances(X):
    n = X.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = np.linalg.norm(X[i] - X[j])
            distances[j, i] = distances[i, j]
    return distances

def hierarchical_clustering(X, n_clusters):
    """
    Implements agglomerative hierarchical clustering with single linkage.
    """
    n = X.shape[0]
    clusters = [[i] for i in range(n)]
    distances = compute_distances(X)

    # Large value to prevent clustering with self
    np.fill_diagonal(distances, np.inf)

    while len(clusters) > n_clusters:
        # Find closest clusters (single linkage)
        min_dist = np.inf
        merge_indices = (-1, -1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = np.min([distances[p, q] for p in clusters[i] for q in clusters[j]])
                if dist < min_dist:
                    min_dist = dist
                    merge_indices = (i, j)

        # Merge clusters
        c1, c2 = merge_indices
        clusters[c1].extend(clusters[c2])
        clusters.pop(c2)

    # Create labels
    labels = np.zeros(n, dtype=int)
    for cluster_id, cluster_pts in enumerate(clusters):
        for pt in cluster_pts:
            labels[pt] = cluster_id

    return labels

if __name__ == "__main__":
    print("Testing Hierarchical Clustering Component...")
    # Synthetic data: 3 clusters
    np.random.seed(42)
    X1 = np.random.normal(loc=[0, 0], scale=0.5, size=(20, 2))
    X2 = np.random.normal(loc=[5, 5], scale=0.5, size=(20, 2))
    X3 = np.random.normal(loc=[10, 0], scale=0.5, size=(20, 2))
    X = np.vstack([X1, X2, X3])

    try:
        labels = hierarchical_clustering(X, n_clusters=3)

        # Verify that the cluster assignments for each generated group are consistent
        assert len(np.unique(labels[:20])) == 1, "Cluster 1 points should have the same label"
        assert len(np.unique(labels[20:40])) == 1, "Cluster 2 points should have the same label"
        assert len(np.unique(labels[40:])) == 1, "Cluster 3 points should have the same label"

        print("Data shape:", X.shape)
        print("Unique labels found:", np.unique(labels))
        print("Cluster assignments for first 10 points:", labels[:10])
        print("Hierarchical clustering successful!")
    except Exception as e:
        print("Failure:", e)
