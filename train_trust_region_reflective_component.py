import numpy as np

def rosenbrock(x):
    """Rosenbrock function."""
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosenbrock_grad(x):
    """Gradient of the Rosenbrock function."""
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    for i in range(1, len(x) - 1):
        grad[i] = 200 * (x[i] - x[i-1]**2) - 400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
    grad[-1] = 200 * (x[-1] - x[-2]**2)
    return grad

def rosenbrock_hess(x):
    """Hessian of the Rosenbrock function."""
    n = len(x)
    H = np.zeros((n, n))
    H[0, 0] = -400 * (x[1] - 3 * x[0]**2) + 2
    H[0, 1] = -400 * x[0]
    for i in range(1, n - 1):
        H[i, i-1] = -400 * x[i-1]
        H[i, i] = 200 - 400 * (x[i+1] - 3 * x[i]**2) + 2
        H[i, i+1] = -400 * x[i]
    H[-1, -2] = -400 * x[-2]
    H[-1, -1] = 200
    return H

def trr_optimize(func, grad, hess, x0, bounds, max_iter=1000, initial_trust_radius=1.0, max_trust_radius=10.0, eta=0.15):
    """
    Simplified Trust Region Reflective (TRF) algorithm for bound constrained optimization.
    """
    x = np.array(x0, dtype=float)
    trust_radius = initial_trust_radius

    n = len(x)
    lower_bounds, upper_bounds = np.array(bounds).T

    for i in range(max_iter):
        g = grad(x)
        H = hess(x)

        # Scale the gradient and Hessian (simplified, typically involves affine scaling)
        v = np.zeros(n)
        for j in range(n):
            if g[j] < 0 and upper_bounds[j] < np.inf:
                v[j] = x[j] - upper_bounds[j]
            elif g[j] > 0 and lower_bounds[j] > -np.inf:
                v[j] = x[j] - lower_bounds[j]
            else:
                v[j] = 1.0 # no bound constraint in this direction

        D = np.diag(np.abs(v)**0.5)
        D_inv = np.diag(1.0 / (np.abs(v)**0.5 + 1e-8))

        g_hat = D @ g
        H_hat = D @ H @ D

        # Solve the trust-region subproblem approximately using a 2D subspace (dogleg-like or Steihaug)
        # Here we use a very simplified approach: find Newton step, find Cauchy step, interpolate

        try:
             p_newton = -np.linalg.solve(H_hat, g_hat)
        except np.linalg.LinAlgError:
             p_newton = -g_hat

        p_cauchy = - (np.dot(g_hat, g_hat) / (np.dot(g_hat, np.dot(H_hat, g_hat)) + 1e-8)) * g_hat

        if np.linalg.norm(p_newton) <= trust_radius:
            p_hat = p_newton
        elif np.linalg.norm(p_cauchy) >= trust_radius:
            p_hat = trust_radius * p_cauchy / np.linalg.norm(p_cauchy)
        else:
            # Simple interpolation (Dogleg)
            tau = (trust_radius - np.linalg.norm(p_cauchy)) / (np.linalg.norm(p_newton - p_cauchy) + 1e-8)
            p_hat = p_cauchy + tau * (p_newton - p_cauchy)

        p = D @ p_hat

        # Reflective step: if p hits a bound, reflect it
        alpha = 1.0
        for j in range(n):
            if p[j] > 0 and x[j] + p[j] > upper_bounds[j]:
                alpha = min(alpha, (upper_bounds[j] - x[j]) / p[j])
            elif p[j] < 0 and x[j] + p[j] < lower_bounds[j]:
                alpha = min(alpha, (lower_bounds[j] - x[j]) / p[j])

        p_reflected = alpha * p
        # simplified reflection: we just truncate to bounds in this basic implementation to guarantee feasibility

        x_new = np.clip(x + p_reflected, lower_bounds, upper_bounds)

        # Evaluate ratio of actual to predicted reduction
        actual_reduction = func(x) - func(x_new)
        predicted_reduction = -np.dot(g, p_reflected) - 0.5 * np.dot(p_reflected, np.dot(H, p_reflected))

        rho = actual_reduction / (predicted_reduction + 1e-8)

        if rho > eta:
            x = x_new
            if rho > 0.75:
                trust_radius = min(2 * trust_radius, max_trust_radius)
        else:
            trust_radius *= 0.25

        if np.linalg.norm(g) < 1e-5:
            break

    return x, func(x)

if __name__ == "__main__":
    print("Testing Trust Region Reflective component on bounded Rosenbrock function...")
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    x0 = np.array([-1.2, 1.0])

    x_opt, f_opt = trr_optimize(rosenbrock, rosenbrock_grad, rosenbrock_hess, x0, bounds)
    print(f"Optimal x: {x_opt}")
    print(f"Optimal f(x): {f_opt}")

    assert np.allclose(x_opt, [1.0, 1.0], atol=1e-1), "TRR failed to converge near global minimum."
    print("Trust Region Reflective component test passed successfully.")
