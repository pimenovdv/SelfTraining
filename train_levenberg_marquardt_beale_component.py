import numpy as np

def beale_residuals(x):
    """
    Residuals for the Beale function.
    f(x, y) = (1.5 - x + x*y)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2
    Min at (3, 0.5) with f = 0
    """
    return np.array([
        1.5 - x[0] + x[0] * x[1],
        2.25 - x[0] + x[0] * x[1]**2,
        2.625 - x[0] + x[0] * x[1]**3
    ])

def beale_jacobian(x):
    """Jacobian of the Beale function residuals."""
    return np.array([
        [-1 + x[1], x[0]],
        [-1 + x[1]**2, 2 * x[0] * x[1]],
        [-1 + x[1]**3, 3 * x[0] * x[1]**2]
    ])

def levenberg_marquardt(residual_func, jacobian_func, x0, tol=1e-8, max_iter=2000):
    x = np.array(x0, dtype=float)
    lam = 1.0
    v = 2.0

    for i in range(max_iter):
        r = residual_func(x)
        J = jacobian_func(x)

        cost = np.sum(r**2)
        if cost < tol:
            print(f"Converged in {i} iterations.")
            break

        JtJ = J.T @ J
        Jtr = J.T @ r

        try:
            diag_elements = np.diag(JtJ)
            diag_elements = np.where(diag_elements < 1e-8, 1e-8, diag_elements)
            delta = np.linalg.solve(JtJ + lam * np.diag(diag_elements), -Jtr)
        except np.linalg.LinAlgError:
            lam *= v
            continue

        x_new = x + delta
        r_new = residual_func(x_new)
        cost_new = np.sum(r_new**2)

        predicted_reduction = -np.dot(delta, Jtr) - 0.5 * np.dot(delta, JtJ @ delta)

        if predicted_reduction > 0:
            rho = (cost - cost_new) / predicted_reduction
        else:
            rho = -1

        if rho > 0:
            x = x_new
            lam = lam * max(1 / 3, 1 - (2 * rho - 1)**3)
            v = 2.0
        else:
            lam *= v
            v *= 2.0

    return x, cost

if __name__ == "__main__":
    print("Testing Levenberg-Marquardt algorithm on Beale function...")
    x0 = np.array([1.0, 1.0])

    x_opt, final_cost = levenberg_marquardt(beale_residuals, beale_jacobian, x0)

    print(f"Optimal point: {x_opt}")
    print(f"Final cost: {final_cost}")

    assert np.allclose(x_opt, [3.0, 0.5], atol=1e-4), "Failed to find minimum of Beale function."
    print("Test passed successfully.")
