import numpy as np

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        clusters = {i: [i] for i in range(n_samples)}

        # Precompute pairwise distances
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = np.linalg.norm(X[i] - X[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        current_clusters = list(clusters.keys())

        while len(current_clusters) > self.n_clusters:
            min_dist = np.inf
            merge_pair = None

            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    c1 = current_clusters[i]
                    c2 = current_clusters[j]

                    if self.linkage == 'single':
                        dist = np.min([dist_matrix[p1, p2] for p1 in clusters[c1] for p2 in clusters[c2]])
                    elif self.linkage == 'complete':
                        dist = np.max([dist_matrix[p1, p2] for p1 in clusters[c1] for p2 in clusters[c2]])
                    elif self.linkage == 'average':
                        dist = np.mean([dist_matrix[p1, p2] for p1 in clusters[c1] for p2 in clusters[c2]])
                    else:
                        raise ValueError("Unsupported linkage")

                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (c1, c2)

            c1, c2 = merge_pair
            clusters[c1].extend(clusters[c2])
            del clusters[c2]
            current_clusters.remove(c2)

        self.labels_ = np.zeros(n_samples, dtype=int)
        for label, (cluster_id, points) in enumerate(clusters.items()):
            for point in points:
                self.labels_[point] = label

        return self

if __name__ == "__main__":
    print("Testing Agglomerative Clustering...")
    np.random.seed(42)
    X1 = np.random.randn(20, 2) + np.array([5, 5])
    X2 = np.random.randn(20, 2) + np.array([-5, -5])
    X = np.vstack((X1, X2))

    true_labels = np.array([0]*20 + [1]*20)

    model = AgglomerativeClustering(n_clusters=2, linkage='single')
    model.fit(X)

    labels = model.labels_

    acc1 = np.mean(labels == true_labels)
    acc2 = np.mean(labels == (1 - true_labels))
    accuracy = max(acc1, acc2)

    print(f"Clustering Accuracy: {accuracy * 100:.2f}%")
    if accuracy == 1.0:
        print("Agglomerative Clustering mathematical evaluation successful.")
    else:
        print("Agglomerative Clustering failed to perfectly separate the clear clusters.")
