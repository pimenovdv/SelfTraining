import numpy as np

def rosenbrock(x):
    """Rosenbrock function."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_residual(x):
    """Residuals of the Rosenbrock function."""
    return np.array([1 - x[0], 10 * (x[1] - x[0]**2)])

def rosenbrock_jacobian(x):
    """Jacobian of the residuals of the Rosenbrock function."""
    return np.array([
        [-1, 0],
        [-20 * x[0], 10]
    ])

def levenberg_marquardt(residual_func, jacobian_func, x0, tol=1e-8, max_iter=2000):
    """
    Levenberg-Marquardt optimization algorithm.
    """
    x = np.array(x0, dtype=float)
    lam = 1.0  # Initial damping parameter
    v = 2.0    # Factor for adjusting damping

    for i in range(max_iter):
        r = residual_func(x)
        J = jacobian_func(x)

        # Current cost
        cost = np.sum(r**2)

        if cost < tol:
            print(f"Converged in {i} iterations.")
            break

        # Normal equations: (J^T J + lam * diag(J^T J)) * delta = -J^T r
        JtJ = J.T @ J
        Jtr = J.T @ r

        # Compute delta with current damping
        try:
            diag_elements = np.diag(JtJ)
            # Avoid singular matrices by adding a small constant if diagonal is zero
            diag_elements = np.where(diag_elements < 1e-8, 1e-8, diag_elements)
            delta = np.linalg.solve(JtJ + lam * np.diag(diag_elements), -Jtr)
        except np.linalg.LinAlgError:
            # If singular, increase damping
            lam *= v
            continue

        # Evaluate new point
        x_new = x + delta
        r_new = residual_func(x_new)
        cost_new = np.sum(r_new**2)

        # Gain ratio (actual reduction / predicted reduction)
        # Predicted reduction = -delta^T J^T r - 0.5 * delta^T J^T J delta
        predicted_reduction = -np.dot(delta, Jtr) - 0.5 * np.dot(delta, JtJ @ delta)

        if predicted_reduction > 0:
            rho = (cost - cost_new) / predicted_reduction
        else:
            rho = -1

        # Update damping and state
        if rho > 0:
            x = x_new
            lam = lam * max(1 / 3, 1 - (2 * rho - 1)**3)
            v = 2.0
        else:
            lam *= v
            v *= 2.0

    return x, cost

if __name__ == "__main__":
    print("Testing Levenberg-Marquardt algorithm on Rosenbrock function...")

    # Test on Rosenbrock function
    x0 = np.array([-1.2, 1.0])

    print(f"Initial point: {x0}")
    print(f"Initial cost: {rosenbrock(x0)}")

    x_opt, final_cost = levenberg_marquardt(rosenbrock_residual, rosenbrock_jacobian, x0)

    print(f"\nOptimization Results:")
    print(f"Optimal point: {x_opt}")
    print(f"Final cost: {final_cost}")
    print(f"True minimum: [1. 1.]")

    # Assertion for CI/CD
    assert np.allclose(x_opt, [1.0, 1.0], atol=1e-4), "Optimization failed to find the true minimum."
    print("Test passed successfully.")
