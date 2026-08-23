import numpy as np

def rbf_kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distance / bandwidth) ** 2)

class MeanShift:
    def __init__(self, bandwidth=2.0, max_iter=300, tol=1e-3):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        centroids = np.copy(X)
        for i in range(self.max_iter):
            new_centroids = []
            for centroid in centroids:
                distances = np.linalg.norm(X - centroid, axis=1)
                weights = rbf_kernel(distances, self.bandwidth)
                new_centroid = np.sum(X * weights[:, np.newaxis], axis=0) / np.sum(weights)
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids)
            if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < self.tol:
                centroids = new_centroids
                break
            centroids = new_centroids

        unique_centroids = []
        labels = []
        for centroid in centroids:
            is_unique = True
            for j, unique_centroid in enumerate(unique_centroids):
                if np.linalg.norm(centroid - unique_centroid) < 1e-1:
                    is_unique = False
                    labels.append(j)
                    break
            if is_unique:
                unique_centroids.append(centroid)
                labels.append(len(unique_centroids) - 1)

        self.cluster_centers_ = np.array(unique_centroids)
        self.labels_ = np.array(labels)

def main():
    print("Testing Mean Shift Clustering Component...")
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[8, 8], scale=0.5, size=(50, 2))
    X = np.vstack([cluster1, cluster2])

    ms = MeanShift(bandwidth=2.0)
    ms.fit(X)

    print(f"Number of estimated clusters: {len(ms.cluster_centers_)}")
    print(f"Cluster centers:\n{ms.cluster_centers_}")

    if len(ms.cluster_centers_) == 2:
        print("Success: Mean Shift correctly identified 2 clusters.")
    else:
        print("Failure: Mean Shift did not identify the correct number of clusters.")
        exit(1)

if __name__ == "__main__":
    main()
