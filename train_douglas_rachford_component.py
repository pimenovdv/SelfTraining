import numpy as np

def soft_threshold(x, kappa):
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)

def douglas_rachford_lasso(A, b, lambda_reg, gamma, max_iter=1000, tol=1e-5):
    """
    Minimizes 1/2 * ||Ax - b||_2^2 + lambda_reg * ||x||_1
    using Douglas-Rachford Splitting.

    f(x) = 1/2 * ||Ax - b||_2^2
    g(x) = lambda_reg * ||x||_1
    """
    n, d = A.shape

    # Precompute (I + gamma * A^T A)^{-1}
    I = np.eye(d)
    inv_mat = np.linalg.inv(I + gamma * A.T @ A)
    A_Tb = A.T @ b

    def prox_f(x):
        return inv_mat @ (x + gamma * A_Tb)

    def prox_g(x):
        return soft_threshold(x, gamma * lambda_reg)

    z = np.zeros(d)
    x = np.zeros(d)

    for i in range(max_iter):
        x_new = prox_f(z)
        y_new = prox_g(2 * x_new - z)
        z_new = z + y_new - x_new

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new
        z = z_new

    return x

if __name__ == '__main__':
    np.random.seed(42)
    n, d = 100, 50
    A = np.random.randn(n, d)
    true_x = np.zeros(d)
    true_x[np.random.choice(d, 5, replace=False)] = np.random.randn(5)
    b = A @ true_x + 0.1 * np.random.randn(n)

    lambda_reg = 0.5
    gamma = 1.0

    x_opt = douglas_rachford_lasso(A, b, lambda_reg, gamma)

    mse = np.mean((A @ x_opt - b)**2)
    sparsity = np.sum(np.abs(x_opt) > 1e-4)

    print(f"MSE: {mse:.6f}")
    print(f"Sparsity: {sparsity} non-zero elements")
