import numpy as np
import time

def soft_threshold(x, lambda_val):
    """
    Soft thresholding operator for L1 regularization.
    prox_{\lambda L1}(x) = sign(x) * max(|x| - \lambda, 0)
    """
    return np.sign(x) * np.maximum(np.abs(x) - lambda_val, 0.0)

def fista(X, y, lambda_val, max_iter=1000, tol=1e-4):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
    for solving L1-regularized least squares (Lasso).

    minimize 0.5 * ||Xw - y||^2_2 + \lambda * ||w||_1
    """
    n_samples, n_features = X.shape

    # Calculate Lipschitz constant of the gradient of the smooth part (0.5 * ||Xw - y||^2_2)
    # L = largest eigenvalue of X^T X
    eigvals = np.linalg.eigvalsh(X.T @ X)
    L = eigvals[-1]

    step_size = 1.0 / L

    w = np.zeros(n_features)
    z = w.copy()
    t = 1.0

    for i in range(max_iter):
        w_old = w.copy()

        # Gradient step on the smooth part, evaluated at z
        grad = X.T @ (X @ z - y)
        z_step = z - step_size * grad

        # Proximal step on the non-smooth part (soft thresholding)
        w = soft_threshold(z_step, step_size * lambda_val)

        # Check convergence
        if np.linalg.norm(w - w_old) < tol:
            break

        # Nesterov momentum update
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        momentum = (t - 1.0) / t_next
        z = w + momentum * (w - w_old)
        t = t_next

    return w

def main():
    print("Testing FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) Component...")
    start_time = time.time()

    np.random.seed(42)

    # Generate synthetic sparse dataset
    n_samples = 100
    n_features = 50

    X = np.random.randn(n_samples, n_features)

    # True weights: sparse (only first 5 features are non-zero)
    true_w = np.zeros(n_features)
    true_w[:5] = [5.0, -3.0, 2.0, -1.0, 4.0]

    # Generate labels with noise
    y = X @ true_w + np.random.randn(n_samples) * 0.1

    # Run FISTA
    lambda_val = 5.0  # Increased lambda for more sparsity
    w_est = fista(X, y, lambda_val)

    # Evaluate sparsity and error
    mse = np.mean((X @ w_est - y)**2)
    sparsity = np.sum(w_est == 0)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Sparsity: {sparsity}/{n_features} weights are exactly zero")
    print(f"Estimated weights (first 10): {w_est[:10]}")
    print(f"True weights (first 10): {true_w[:10]}")

    exec_time = time.time() - start_time
    print(f"Execution time: {exec_time:.4f} seconds")

    if sparsity > 30 and mse < 0.5:
        print("FISTA component successfully optimized the L1-regularized objective.")
    else:
        print("Warning: FISTA optimization may have failed.")

if __name__ == "__main__":
    main()
