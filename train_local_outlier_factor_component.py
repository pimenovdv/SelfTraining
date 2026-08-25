import numpy as np

def compute_distances(X):
    sq_X = np.sum(X**2, axis=1)
    dists = np.sqrt(np.maximum(sq_X[:, None] + sq_X[None, :] - 2 * np.dot(X, X.T), 0))
    return dists

class LocalOutlierFactor:
    def __init__(self, k=5):
        self.k = k

    def fit_predict(self, X):
        dists = compute_distances(X)
        np.fill_diagonal(dists, np.inf)

        # Find k-nearest neighbors and their distances (k-distance)
        k_dists = np.sort(dists, axis=1)[:, self.k - 1]

        # Reachability distance
        reach_dists = np.maximum(dists, k_dists)

        # Local reachability density (lrd)
        knn_indices = np.argsort(dists, axis=1)[:, :self.k]

        lrd = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            reach_sum = np.sum(reach_dists[i, knn_indices[i]])
            lrd[i] = self.k / reach_sum if reach_sum > 0 else np.inf

        # Local outlier factor (LOF)
        lof = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            lrd_ratio_sum = np.sum(lrd[knn_indices[i]] / lrd[i])
            lof[i] = lrd_ratio_sum / self.k

        return lof

if __name__ == "__main__":
    print("Testing Local Outlier Factor (LOF) Component...")

    np.random.seed(42)
    # Normal data
    X_normal = np.random.randn(50, 2)
    # Outliers
    X_outliers = np.array([[5, 5], [-5, -5], [5, -5], [-5, 5]])

    X = np.vstack((X_normal, X_outliers))

    model = LocalOutlierFactor(k=5)
    lof_scores = model.fit_predict(X)

    # Check if outliers have higher LOF scores
    normal_lof_mean = np.mean(lof_scores[:50])
    outlier_lof_mean = np.mean(lof_scores[50:])

    print(f"Mean LOF for normal data: {normal_lof_mean:.4f}")
    print(f"Mean LOF for outliers: {outlier_lof_mean:.4f}")

    if outlier_lof_mean > normal_lof_mean * 1.5:
        print("Local Outlier Factor successfully identified anomalies.")
    else:
        print("Local Outlier Factor failed to identify anomalies.")
        exit(1)
