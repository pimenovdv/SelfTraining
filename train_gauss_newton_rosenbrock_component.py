import numpy as np

def rosenbrock_residuals(x):
    """
    Computes the residuals of the Rosenbrock function.
    f(x, y) = 100 * (y - x**2)**2 + (1 - x)**2
    We split it into two residuals:
    r1 = 10 * (y - x**2)
    r2 = 1 - x
    """
    r1 = 10.0 * (x[1] - x[0]**2)
    r2 = 1.0 - x[0]
    return np.array([r1, r2])

def rosenbrock_jacobian(x):
    """
    Computes the Jacobian matrix of the Rosenbrock residuals.
    J = [dr1/dx, dr1/dy]
        [dr2/dx, dr2/dy]
    dr1/dx = -20 * x
    dr1/dy = 10
    dr2/dx = -1
    dr2/dy = 0
    """
    J = np.zeros((2, 2))
    J[0, 0] = -20.0 * x[0]
    J[0, 1] = 10.0
    J[1, 0] = -1.0
    J[1, 1] = 0.0
    return J

class GaussNewtonOptimizer:
    """
    Gauss-Newton algorithm for non-linear least squares optimization.
    Minimizes S(x) = 0.5 * sum(r_i(x)**2)
    """
    def __init__(self, tol=1e-6, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter

    def optimize(self, x0, residuals_fn, jacobian_fn):
        x = np.array(x0, dtype=float)
        history = [x.copy()]

        for i in range(self.max_iter):
            r = residuals_fn(x)
            J = jacobian_fn(x)

            # Normal equations: (J^T J) dx = -J^T r
            # Solve for dx: dx = -inv(J^T J) J^T r
            # Using pseudo-inverse or linear solver for stability
            J_t_J = J.T @ J
            J_t_r = J.T @ r

            try:
                dx = np.linalg.solve(J_t_J, -J_t_r)
            except np.linalg.LinAlgError:
                # Fallback if singular
                dx = np.linalg.pinv(J_t_J) @ (-J_t_r)

            x_new = x + dx
            history.append(x_new.copy())

            if np.linalg.norm(dx) < self.tol:
                break

            x = x_new

        return x, np.array(history)

if __name__ == "__main__":
    # Test Gauss-Newton on Rosenbrock
    optimizer = GaussNewtonOptimizer()

    # Starting point far from minimum (1, 1)
    x0 = [-1.0, 2.0]

    final_x, history = optimizer.optimize(x0, rosenbrock_residuals, rosenbrock_jacobian)

    # Reconstruct final loss
    final_r = rosenbrock_residuals(final_x)
    final_loss = 0.5 * np.sum(final_r**2)

    print("Initial parameters:", x0)
    print("Final parameters:", final_x)
    print("Final loss:", final_loss)
    print("Iterations:", len(history) - 1)

    assert np.allclose(final_x, [1.0, 1.0], atol=1e-4), "Failed to find global minimum"
    assert final_loss < 1e-8, "Final loss is too high"
    print("Optimization finished successfully.")
