import numpy as np

def soft_threshold(rho, alpha):
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0

def coordinate_descent_lasso(X, y, alpha, num_iters=1000, tol=1e-4):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    for iteration in range(num_iters):
        w_old = w.copy()
        for j in range(n_features):
            w_temp = w[j]
            w[j] = 0

            r_j = y - np.dot(X, w)
            rho_j = np.dot(X[:, j], r_j)
            z_j = np.dot(X[:, j], X[:, j])

            w[j] = soft_threshold(rho_j, alpha) / z_j

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_w = np.array([1.5, -2.0, 0.0, 0.0, 3.2])
    y = np.dot(X, true_w) + np.random.randn(100) * 0.1

    alpha = 1.0 # L1 penalty
    w_est = coordinate_descent_lasso(X, y, alpha=alpha)

    print("True weights:", true_w)
    print("Estimated weights:", w_est)

    loss = 0.5 * np.sum((y - np.dot(X, w_est))**2) + alpha * np.sum(np.abs(w_est))
    print(f"Final Lasso Objective: {loss:.4f}")
