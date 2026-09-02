import numpy as np

def rbf_kernel(X, Y, gamma=1.0):
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * sq_dists)

def train_lpp(X, n_components=2, k_neighbors=5, gamma=1.0):
    N = X.shape[0]
    # Compute pairwise squared distances
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)

    # K-nearest neighbors graph
    W = np.zeros((N, N))
    for i in range(N):
        idx = np.argsort(sq_dists[i])[:k_neighbors+1]
        for j in idx:
            if i != j:
                W[i, j] = np.exp(-gamma * sq_dists[i, j])
                W[j, i] = W[i, j] # symmetric

    D = np.diag(np.sum(W, axis=1))
    L = D - W

    # Solve generalized eigenvalue problem: X L X^T a = lambda X D X^T a
    XLX = X.T @ L @ X
    XDX = X.T @ D @ X

    # Regularization
    XDX += 1e-5 * np.eye(XDX.shape[0])

    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(XDX) @ XLX)

    # Sort eigenvalues
    idx = np.argsort(eigvals.real)
    components = eigvecs[:, idx[:n_components]].real

    return components

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(100, 10)

    components = train_lpp(X, n_components=2)
    print(f"LPP trained successfully. Transformation matrix shape: {components.shape}")
