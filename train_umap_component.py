import numpy as np

def compute_distances(X):
    # compute pairwise euclidean distances
    n = X.shape[0]
    sq_dists = np.sum(X**2, axis=1).reshape(-1,1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    # avoid negative numbers due to float precision
    return np.sqrt(np.maximum(sq_dists, 0))

def compute_local_fuzzy_simplicial_set(distances, n_neighbors=15):
    n = distances.shape[0]
    sigmas = np.zeros(n)
    rhos = np.zeros(n)

    # We simply use a heuristic for sigma here
    for i in range(n):
        dist_i = np.sort(distances[i])
        rhos[i] = dist_i[1] # distance to first nearest neighbor (idx 0 is self)

        # Binary search for sigma to match log(k) entropy
        # Simplified for demonstration
        sigmas[i] = dist_i[n_neighbors] - rhos[i] if dist_i[n_neighbors] > rhos[i] else 1.0

    # Compute graph weights
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = distances[i, j]
                if d <= rhos[i]:
                    graph[i, j] = 1.0
                else:
                    graph[i, j] = np.exp(-(d - rhos[i]) / (sigmas[i] + 1e-8))

    # Symmetrize
    symmetrized = graph + graph.T - graph * graph.T
    return symmetrized

def compute_umap(X, n_epochs=50, n_neighbors=5, min_dist=0.1, n_components=2):
    distances = compute_distances(X)
    graph = compute_local_fuzzy_simplicial_set(distances, n_neighbors)

    n = X.shape[0]
    # Initialize embeddings (e.g., random or spectral)
    Y = np.random.normal(scale=0.1, size=(n, n_components))

    # A simplified repulsive/attractive gradient descent
    # For UMAP, we typically use Cross Entropy
    lr = 1.0
    for epoch in range(n_epochs):
        grad_Y = np.zeros_like(Y)

        # compute distances in low dim
        dist_Y = compute_distances(Y)

        # Simplified UMAP gradient calculation (Approximation of Force Directed Graph)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = dist_Y[i,j]**2

                    # Attractive force
                    p_ij = graph[i,j]

                    # This is a very simplified repulsive/attractive formulation
                    # for mathematical demonstration purposes without external libraries
                    diff = Y[i] - Y[j]

                    # Attractive term
                    attr = p_ij * diff / (1.0 + d)

                    # Repulsive term (sampling approximation is usually done, here we do all pairs)
                    rep = (1.0 - p_ij) * diff / ((0.001 + d) * (1.0 + d))

                    grad_Y[i] += attr - rep

        Y -= lr * grad_Y

    return Y

def main():
    np.random.seed(42)
    # Synthetic data: 2 clusters
    X1 = np.random.normal(loc=0, scale=0.5, size=(10, 5))
    X2 = np.random.normal(loc=5, scale=0.5, size=(10, 5))
    X = np.vstack([X1, X2])

    Y = compute_umap(X, n_epochs=10)
    print("UMAP mathematically simulated on 20 points.")
    print("Shape of reduced data:", Y.shape)

if __name__ == "__main__":
    main()
