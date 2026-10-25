import numpy as np
import time

def conjugate_gradient(A, b, x0=None, tol=1e-5, max_iter=1000):
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
    r = b - A @ x
    p = r.copy()
    rsold = r.T @ r

    losses = []

    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.T @ r

        loss = np.sqrt(np.sum(r**2))
        losses.append(loss)

        if np.sqrt(rsnew) < tol:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, losses

if __name__ == "__main__":
    print("Training Conjugate Gradient Component...")

    # Generate random symmetric positive-definite matrix
    np.random.seed(42)
    N = 100
    M = np.random.randn(N, N)
    A = M.T @ M + np.eye(N) * 0.1

    # Generate true solution
    x_true = np.random.randn(N)
    b = A @ x_true

    start_time = time.time()
    x_pred, losses = conjugate_gradient(A, b, tol=1e-8)
    end_time = time.time()

    mse = np.mean((x_pred - x_true)**2)

    print(f"Iterations: {len(losses)}")
    print(f"Final Residual Norm: {losses[-1]:.6e}")
    print(f"Mean Squared Error vs True x: {mse:.6e}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")

    if mse < 1e-6:
        print("Conjugate Gradient converged successfully.")
    else:
        print("Conjugate Gradient failed to converge.")
        exit(1)
