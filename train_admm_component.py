import numpy as np

def admm_lasso(A, b, rho, alpha, max_iters=100, tol=1e-4):
    """
    Alternating Direction Method of Multipliers (ADMM) for Lasso:
    min 1/2 * ||Ax - b||_2^2 + alpha * ||z||_1
    s.t. x - z = 0
    """
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    # Precompute A^T A and A^T b for the x-update
    AtA = A.T @ A
    Atb = A.T @ b

    # Cache the matrix factorization for efficiency
    inv_mat = np.linalg.inv(AtA + rho * np.eye(n))

    for i in range(max_iters):
        # x-update: x = (A^T A + rho * I)^(-1) * (A^T b + rho * (z - u))
        x = inv_mat @ (Atb + rho * (z - u))

        # z-update: soft-thresholding
        kappa = alpha / rho
        v = x + u
        z_new = np.sign(v) * np.maximum(0, np.abs(v) - kappa)

        # u-update
        u = u + x - z_new

        # Convergence check
        if np.linalg.norm(z_new - z) < tol and np.linalg.norm(x - z_new) < tol:
            z = z_new
            break

        z = z_new

    return z

if __name__ == "__main__":
    # Generate synthetic data for Lasso
    np.random.seed(42)
    m, n = 50, 100
    A = np.random.randn(m, n)
    x_true = np.zeros(n)
    # create sparse x
    x_true[np.random.choice(n, 10, replace=False)] = np.random.randn(10)
    b = A @ x_true + 0.1 * np.random.randn(m)

    rho = 1.0
    alpha = 0.5

    x_pred = admm_lasso(A, b, rho, alpha)

    mse = np.mean((x_pred - x_true)**2)
    sparsity = np.sum(np.abs(x_pred) > 1e-5)

    print(f"ADMM Lasso MSE: {mse:.6f}")
    print(f"Sparsity: {sparsity} non-zero elements")
