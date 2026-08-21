import numpy as np

def isomap(X, n_neighbors=5, n_components=2):
    n_samples = X.shape[0]
    dist_sq = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    dist = np.sqrt(np.maximum(dist_sq, 0))
    graph = np.full((n_samples, n_samples), np.inf)
    for i in range(n_samples):
        indices = np.argsort(dist[i])[:n_neighbors+1]
        graph[i, indices] = dist[i, indices]
        graph[indices, i] = dist[i, indices]
    for k in range(n_samples):
        for i in range(n_samples):
            for j in range(n_samples):
                if graph[i, j] > graph[i, k] + graph[k, j]:
                    graph[i, j] = graph[i, k] + graph[k, j]
    D2 = graph**2
    J = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    B = -0.5 * J.dot(D2).dot(J)
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]
    Y = eigenvectors.dot(np.diag(np.sqrt(np.maximum(eigenvalues, 0))))
    return Y

if __name__ == "__main__":
    print("Testing Isomap Component...")
    np.random.seed(42)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(100))
    x = t * np.cos(t)
    y = 21 * np.random.rand(100)
    z = t * np.sin(t)
    X = np.column_stack((x, y, z))
    try:
        Y = isomap(X, n_neighbors=10, n_components=2)
        print("Original Shape:", X.shape)
        print("Reduced Shape:", Y.shape)
        print("Success")
    except Exception as e:
        print("Failure:", e)
