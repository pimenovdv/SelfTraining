import numpy as np

def affinity_propagation(S, max_iter=200, damping=0.5):
    n = S.shape[0]
    R = np.zeros((n, n))
    A = np.zeros((n, n))

    for _ in range(max_iter):
        # Update responsibility
        AS = A + S
        for i in range(n):
            for k in range(n):
                idx = np.arange(n) != k
                max_val = np.max(AS[i, idx])
                R[i, k] = damping * R[i, k] + (1 - damping) * (S[i, k] - max_val)

        # Update availability
        for i in range(n):
            for k in range(n):
                if i == k:
                    idx = np.arange(n) != k
                    A[k, k] = damping * A[k, k] + (1 - damping) * np.sum(np.maximum(0, R[idx, k]))
                else:
                    idx = (np.arange(n) != k) & (np.arange(n) != i)
                    A[i, k] = damping * A[i, k] + (1 - damping) * min(0, R[k, k] + np.sum(np.maximum(0, R[idx, k])))

    exemplars = np.argmax(A + R, axis=1)
    return exemplars

if __name__ == "__main__":
    np.random.seed(42)
    X = np.array([1, 2, 3, 10, 11, 12, 20, 21, 22], dtype=float)
    n = len(X)
    S = -np.square(X[:, None] - X[None, :])
    np.fill_diagonal(S, np.median(S))

    print("Running Affinity Propagation...")
    labels = affinity_propagation(S, max_iter=50, damping=0.5)
    print("Cluster assignments:", labels)

    unique_clusters = np.unique(labels)
    print("Number of clusters:", len(unique_clusters))
    if len(unique_clusters) == 3:
        print("Success! Affinity Propagation found the correct number of clusters.")
    else:
        print("Failed to find the correct number of clusters.")
        exit(1)
