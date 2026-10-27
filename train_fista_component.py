import numpy as np

def fista(A, b, lambda_reg, max_iters=1000, tol=1e-5):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
    Minimizes: 0.5 * ||Ax - b||_2^2 + lambda * ||x||_1
    """
    m, n = A.shape
    x = np.zeros(n)
    y = np.zeros(n)
    t = 1.0

    L = np.linalg.norm(A.T @ A, 2)
    step_size = 1.0 / L

    for i in range(max_iters):
        x_old = x.copy()

        # Gradient step
        grad = A.T @ (A @ y - b)
        z = y - step_size * grad

        # Proximal step (Soft thresholding)
        threshold = lambda_reg * step_size
        x = np.sign(z) * np.maximum(0, np.abs(z) - threshold)

        # Momentum step
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next

        # Convergence check
        if np.linalg.norm(x - x_old) < tol:
            break

    return x

if __name__ == "__main__":
    np.random.seed(42)
    m, n = 50, 100
    A = np.random.randn(m, n)
    x_true = np.zeros(n)
    # create sparse x
    x_true[np.random.choice(n, 10, replace=False)] = np.random.randn(10)
    b = A @ x_true + 0.1 * np.random.randn(m)

    lambda_reg = 1.0
    x_pred = fista(A, b, lambda_reg)

    mse = np.mean((x_pred - x_true)**2)
    sparsity = np.sum(x_pred != 0)

    print(f"FISTA MSE: {mse:.6f}")
    print(f"Sparsity: {sparsity} non-zero elements")
