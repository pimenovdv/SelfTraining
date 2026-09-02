import numpy as np

def rbf_kernel(X, Y, gamma):
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * sq_dists)

def train_rvm(X, y, n_iter=100, alpha_tol=1e9):
    N, D = X.shape
    PHI = rbf_kernel(X, X, gamma=0.1)

    alpha = np.ones(N) * 1e-6
    beta = 1.0

    for i in range(n_iter):
        A = np.diag(alpha)
        Sigma = np.linalg.inv(A + beta * PHI.T @ PHI)
        mu = beta * Sigma @ PHI.T @ y

        gamma = 1 - alpha * np.diag(Sigma)

        alpha_new = gamma / (mu**2 + 1e-10)

        # Prune irrelevant vectors
        keep = alpha_new < alpha_tol

        if not np.any(keep):
            break

        alpha_new[~keep] = alpha_tol

        alpha = alpha_new

        err = y - PHI @ mu
        beta = (N - np.sum(gamma)) / (np.sum(err**2) + 1e-10)

    return keep, mu, beta

if __name__ == "__main__":
    np.random.seed(42)
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = np.sinc(X).flatten() + np.random.normal(0, 0.1, 100)

    keep, mu, beta = train_rvm(X, y)
    print(f"RVM trained successfully.")
    print(f"Number of Relevance Vectors: {np.sum(keep)}")
