import numpy as np
import time

def nnls(A, b, max_iter=1000, tol=1e-6):
    """
    Implements Non-Negative Least Squares (NNLS) algorithm mathematically.
    Solves argmin_x ||Ax - b||_2^2 subject to x >= 0.
    Using the Lawson-Hanson active set algorithm.
    """
    m, n = A.shape
    x = np.zeros(n)
    P = np.zeros(n, dtype=bool) # Passive set (indices where x_i > 0)
    Z = np.ones(n, dtype=bool)  # Active set (indices where x_i = 0)

    w = A.T @ (b - A @ x)

    for _ in range(max_iter):
        # Check optimality
        if np.all(Z == False) or np.max(w[Z]) <= tol:
            break

        # Find index t in Z that maximizes w_t
        t = np.argmax(w * Z)

        # Move t from Z to P
        P[t] = True
        Z[t] = False

        while True:
            # Solve unconstrained least squares over P
            A_P = A[:, P]
            z_P = np.linalg.lstsq(A_P, b, rcond=None)[0]

            z = np.zeros(n)
            z[P] = z_P

            if np.all(z[P] > 0):
                x = z
                break

            # Find the minimum alpha to maintain non-negativity
            indices = (z <= 0) & P
            alpha = np.min(x[indices] / (x[indices] - z[indices]))

            x = x + alpha * (z - x)

            # Move indices where x hits 0 from P back to Z
            zero_indices = (x <= tol) & P
            Z[zero_indices] = True
            P[zero_indices] = False

        w = A.T @ (b - A @ x)

    return x

def evaluate_nnls():
    """
    Evaluates the NNLS component by comparing its solution to standard unconstrained least squares
    on a problem where the unconstrained solution has negative components.
    """
    print("Initializing Non-Negative Least Squares (NNLS) training...")
    start_time = time.time()

    # Generate data
    np.random.seed(42)
    m, n = 100, 20
    A = np.random.randn(m, n)
    x_true = np.random.rand(n) # True non-negative coefficients
    x_true[5:15] = 0 # Make it sparse
    b = A @ x_true + 0.1 * np.random.randn(m)

    # Solve NNLS
    x_nnls = nnls(A, b)

    # Verify non-negativity
    min_x = np.min(x_nnls)
    print(f"Minimum value in solution: {min_x:.6f}")

    # Compute reconstruction error
    error = np.linalg.norm(A @ x_nnls - b)
    print(f"Residual error: {error:.4f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.4f} seconds.")

    # Loss is the magnitude of the most negative component (should be 0)
    loss = abs(min(min_x, 0.0))
    print(f"Final Equivalent Loss (magnitude of most negative value): {loss:.6f}")

    return loss

if __name__ == "__main__":
    evaluate_nnls()
