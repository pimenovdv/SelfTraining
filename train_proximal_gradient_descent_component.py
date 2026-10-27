import numpy as np
import time

def soft_thresholding(x, lambda_val):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_val, 0)

def proximal_gradient_descent(X, y, lambda_val, lr, max_iters):
    m, n = X.shape
    w = np.zeros(n)

    for i in range(max_iters):
        # Gradient of the smooth part (least squares: 0.5 * ||Xw - y||_2^2)
        grad = X.T @ (X @ w - y) / m

        # Gradient descent step
        w_temp = w - lr * grad

        # Proximal step (soft thresholding for L1 penalty)
        w = soft_thresholding(w_temp, lr * lambda_val)

    return w

if __name__ == "__main__":
    print("Testing Proximal Gradient Descent for Lasso...")
    np.random.seed(42)

    # Generate synthetic data
    m, n = 100, 50
    X = np.random.randn(m, n)
    true_w = np.zeros(n)
    true_w[:5] = [2.5, -1.3, 0.8, -0.5, 3.0] # Only 5 non-zero coefficients
    y = X @ true_w + 0.1 * np.random.randn(m)

    lambda_val = 0.5
    lr = 0.05
    max_iters = 500

    start_time = time.time()
    w_hat = proximal_gradient_descent(X, y, lambda_val, lr, max_iters)
    end_time = time.time()

    mse = np.mean((X @ w_hat - y)**2)
    sparsity = np.sum(np.abs(w_hat) < 1e-5)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Sparsity: {sparsity}/{n} zero coefficients")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
