import numpy as np
import time

def frank_wolfe(A, b, tau=1.0, max_iter=1000, tol=1e-5):
    """
    Frank-Wolfe algorithm for solving constrained least squares:
    min 0.5 * ||A x - b||_2^2
    subject to ||x||_1 <= tau
    """
    n_features = A.shape[1]
    x = np.zeros(n_features)

    losses = []

    for t in range(max_iter):
        # Gradient of f(x) = 0.5 * ||Ax - b||^2
        # grad = A^T (Ax - b)
        residual = A.dot(x) - b
        grad = A.T.dot(residual)

        loss = 0.5 * np.sum(residual**2)
        losses.append(loss)

        # Linear minimization oracle (LMO) over L1 ball: ||s||_1 <= tau
        # min_s <s, grad> s.t. ||s||_1 <= tau
        # Solution: s = -tau * sign(grad_i) * e_i, where i = argmax |grad|
        idx = np.argmax(np.abs(grad))
        s = np.zeros(n_features)
        s[idx] = -tau * np.sign(grad[idx])

        # Frank-Wolfe step
        gamma = 2.0 / (t + 2.0)

        x_next = (1.0 - gamma) * x + gamma * s

        if np.linalg.norm(x_next - x) < tol:
            print(f"Converged at iteration {t}")
            break

        x = x_next

    return x, losses

def main():
    print("Starting Frank-Wolfe Algorithm Component Training...")
    start_time = time.time()

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    A = np.random.randn(n_samples, n_features)
    # True x is sparse
    true_x = np.zeros(n_features)
    true_x[0] = 0.8
    true_x[5] = -0.5
    true_x[10] = 0.3

    b = A.dot(true_x) + 0.1 * np.random.randn(n_samples)

    # Solve using Frank-Wolfe
    tau = np.sum(np.abs(true_x)) # Allow sufficient L1 norm
    x_opt, losses = frank_wolfe(A, b, tau=tau, max_iter=500)

    end_time = time.time()

    print(f"Final Loss: {losses[-1]:.6f}")
    print(f"Sparsity of solution (non-zeros): {np.sum(np.abs(x_opt) > 1e-4)}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    main()
