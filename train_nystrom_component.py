import numpy as np

def rbf_kernel(X, Y, gamma=1.0):
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True)
    dist2 = X_sq - 2 * np.dot(X, Y.T) + Y_sq.T
    return np.exp(-gamma * dist2)

def nystrom_approximation(X, n_components, gamma=1.0):
    N = X.shape[0]

    indices = np.random.choice(N, n_components, replace=False)
    landmarks = X[indices]

    C = rbf_kernel(X, landmarks, gamma=gamma)
    W = rbf_kernel(landmarks, landmarks, gamma=gamma)

    U, S, Vt = np.linalg.svd(W)

    tol = 1e-8
    S_inv_sqrt = np.zeros_like(S)
    mask = S > tol
    S_inv_sqrt[mask] = 1.0 / np.sqrt(S[mask])

    W_inv_sqrt = U @ np.diag(S_inv_sqrt) @ Vt

    X_transformed = C @ W_inv_sqrt

    return X_transformed, C, W_inv_sqrt

def main():
    print("Initializing Nyström Method Component...")

    np.random.seed(42)
    N = 1000
    D = 10
    X = np.random.randn(N, D)

    m = 100
    gamma = 0.1

    print(f"Data shape: {X.shape}")
    print(f"Number of landmarks (m): {m}")

    X_transformed, C, W_inv_sqrt = nystrom_approximation(X, m, gamma=gamma)

    print(f"Transformed data shape: {X_transformed.shape}")

    subset_size = 200
    K_exact = rbf_kernel(X[:subset_size], X[:subset_size], gamma=gamma)
    K_approx = X_transformed[:subset_size] @ X_transformed[:subset_size].T

    error = np.linalg.norm(K_exact - K_approx, ord='fro') / np.linalg.norm(K_exact, ord='fro')
    print(f"Relative Frobenius norm error of approximation: {error:.4f}")

    if error < 0.2:
        print("Success: Approximation error is sufficiently low.")
    else:
        print("Warning: Approximation error might be high.")

if __name__ == "__main__":
    main()
