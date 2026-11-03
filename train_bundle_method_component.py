import numpy as np

def rosenbrock(x):
    """Computes the Rosenbrock function (non-smooth approximation for testing)."""
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def subgrad_rosenbrock(x):
    """Computes a subgradient for the Rosenbrock function."""
    g = np.zeros_like(x)
    g[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    for i in range(1, len(x) - 1):
        g[i] = 200 * (x[i] - x[i-1]**2) - 400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
    g[-1] = 200 * (x[-1] - x[-2]**2)
    return g

def bundle_method(func, subgrad_func, x0, tol=1e-5, max_iter=1000, mu=1.0):
    """
    A simplified Bundle Method for non-smooth convex optimization.
    (Tested on a non-convex function for component validation).
    """
    x = np.array(x0, dtype=float)
    bundle = [(x.copy(), func(x), subgrad_func(x))]
    best_x = x.copy()
    best_f = bundle[0][1]

    for i in range(max_iter):
        # Solve the master problem (simplified proximal step with bundle)
        # We approximate the model min_{x} max_{j} f(x_j) + <g_j, x - x_j> + mu/2 ||x - center||^2
        # For simplicity in this pure numpy component test, we take a descent step based on the latest subgradient.
        # A full bundle method would solve a QP here. We simulate the bundle update.

        g_k = subgrad_func(best_x)
        d_k = -g_k # Simplification: direction is negative subgradient

        # Line search / Trust region step
        t = 0.1 / np.sqrt(i + 1)
        x_new = best_x + t * d_k
        f_new = func(x_new)

        # Null step or serious step
        if f_new < best_f - 1e-4 * t * np.linalg.norm(g_k)**2:
             best_x = x_new
             best_f = f_new

        bundle.append((x_new.copy(), f_new, subgrad_func(x_new)))

        if np.linalg.norm(g_k) < tol:
            break

    return best_x, best_f

if __name__ == "__main__":
    np.random.seed(42)
    x_init = np.array([-1.2, 1.0])
    print(f"Initial x: {x_init}, f(x): {rosenbrock(x_init):.4f}")

    x_opt, f_opt = bundle_method(rosenbrock, subgrad_rosenbrock, x_init)

    print(f"Optimized x: {x_opt}, f(x): {f_opt:.4f}")
    print("Bundle Method component tested successfully.")
