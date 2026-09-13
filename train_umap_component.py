import numpy as np

def compute_distances(X):
    # Pairwise squared Euclidean distance
    sum_x = np.sum(np.square(X), 1)
    dist = np.add(np.add(-2 * np.dot(X, X.T), sum_x).T, sum_x)
    return np.sqrt(np.maximum(dist, 0.0))

def find_sigmas_and_rhos(distances, k, tol=1e-5, max_iter=50):
    n_samples = distances.shape[0]
    sigmas = np.zeros(n_samples)
    rhos = np.zeros(n_samples)

    target = np.log2(k)

    for i in range(n_samples):
        dists_i = distances[i]
        # Ignore self distance which is 0
        non_zero_dists = dists_i[dists_i > 0]

        if len(non_zero_dists) > 0:
            rhos[i] = np.min(non_zero_dists)
        else:
            rhos[i] = 0.0

        lo, hi = 0.0, np.inf
        mid = 1.0

        for _ in range(max_iter):
            val = np.sum(np.exp(-np.maximum(0, dists_i - rhos[i]) / mid))

            if np.abs(val - target) < tol:
                break

            if val > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        sigmas[i] = mid

    return sigmas, rhos

def compute_fuzzy_simplicial_set(X, k=15):
    n_samples = X.shape[0]
    distances = compute_distances(X)
    sigmas, rhos = find_sigmas_and_rhos(distances, k)

    P = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                d = distances[i, j]
                P[i, j] = np.exp(-np.maximum(0, d - rhos[i]) / sigmas[i])

    # Symmetrize
    P = P + P.T - P * P.T
    return P

def umap_layout(P, n_components=2, n_epochs=500, a=1.0, b=1.0, lr=1.0):
    n_samples = P.shape[0]
    # Initialize embeddings randomly
    Y = np.random.randn(n_samples, n_components) * 10.0

    # Get positive edges
    sources, targets = np.nonzero(P)

    # Pre-compute edge probabilities for sampling
    p_weights = P[sources, targets]
    p_weights = p_weights / np.max(p_weights)

    for epoch in range(n_epochs):
        for idx in range(len(sources)):
            if np.random.rand() > p_weights[idx]:
                continue

            i, j = sources[idx], targets[idx]

            # Positive pull (attraction)
            # Differentiate: y_i - y_j
            # gradient: 2ab / (1 + a * d^b) * (y_i - y_j)
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff ** 2)
            grad_pos = -2.0 * a * b * (dist_sq ** (b - 1.0)) / (1.0 + a * dist_sq ** b) * diff
            grad_pos = np.clip(grad_pos, -4.0, 4.0)
            Y[i] += lr * grad_pos
            Y[j] -= lr * grad_pos

            # Negative push (repulsion) (sample multiple random nodes)
            n_neg_samples = 5
            for _ in range(n_neg_samples):
                k = np.random.randint(n_samples)
                if k != i:
                    diff_neg = Y[i] - Y[k]
                    dist_sq_neg = np.sum(diff_neg ** 2)

                    grad_neg = -2.0 * b / (0.001 + dist_sq_neg) / (1.0 + a * dist_sq_neg ** b) * diff_neg
                    grad_neg = np.clip(grad_neg, -4.0, 4.0)

                    Y[i] -= lr * grad_neg

    return Y

def test_umap_component():
    np.random.seed(42)
    # Generate simple dataset: two clusters
    X1 = np.random.randn(50, 5) + np.array([5, 5, 5, 5, 5])
    X2 = np.random.randn(50, 5) - np.array([5, 5, 5, 5, 5])
    X = np.vstack([X1, X2])

    print("Computing fuzzy simplicial set...")
    P = compute_fuzzy_simplicial_set(X, k=15)

    print("Computing UMAP layout...")
    # Using default a=1, b=1 which is related to standard t-distribution-like tail
    Y = umap_layout(P, n_components=2, n_epochs=200, lr=1.0)

    # Calculate distances in original space vs embedded space
    d_orig = compute_distances(X)
    d_emb = compute_distances(Y)

    # Intra-cluster distances vs Inter-cluster distances
    # Cluster 1: 0-49, Cluster 2: 50-99
    c1_intra = np.mean(d_emb[:50, :50])
    c2_intra = np.mean(d_emb[50:, 50:])
    inter_c = np.mean(d_emb[:50, 50:])

    print(f"Intra-cluster 1 distance (embedded): {c1_intra:.4f}")
    print(f"Intra-cluster 2 distance (embedded): {c2_intra:.4f}")
    print(f"Inter-cluster distance (embedded): {inter_c:.4f}")

    if inter_c > c1_intra * 2 and inter_c > c2_intra * 2:
        print("Success: UMAP preserved cluster separation!")
    else:
        print("Warning: UMAP did not clearly separate clusters.")

if __name__ == "__main__":
    test_umap_component()
