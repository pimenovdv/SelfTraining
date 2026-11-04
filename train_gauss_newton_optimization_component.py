import numpy as np

def gauss_newton(residual_func, jacobian_func, x0, tol=1e-8, max_iter=100):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        r = residual_func(x)
        J = jacobian_func(x)

        cost = np.sum(r**2)
        if cost < tol:
            print(f"Converged in {i} iterations.")
            break

        JtJ = J.T @ J
        Jtr = J.T @ r

        delta = np.linalg.solve(JtJ, -Jtr)
        x = x + delta

    return x, cost

def test_residual(x):
    # Non-linear exponential model y = a * exp(b * x)
    # Synthetic data
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([2.5, 3.6, 5.3, 7.8, 11.5])
    return y - x[0] * np.exp(x[1] * t)

def test_jacobian(x):
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    J = np.zeros((len(t), 2))
    J[:, 0] = -np.exp(x[1] * t)
    J[:, 1] = -x[0] * t * np.exp(x[1] * t)
    return J

def train_gauss_newton_optimization():
    print("Testing Gauss-Newton algorithm on exponential model...")
    x0 = np.array([1.0, 0.0])
    print(f"Initial point: {x0}")

    x_opt, final_cost = gauss_newton(test_residual, test_jacobian, x0)

    print(f"\nOptimization Results:")
    print(f"Optimal point: {x_opt}")
    print(f"Final cost: {final_cost}")
    print(f"Expected optimal values roughly: [2.5, 0.38]")

    # Check convergence to reasonable values
    assert np.allclose(x_opt, [2.5, 0.38], atol=0.1)
    print("Test passed successfully.")

if __name__ == "__main__":
    train_gauss_newton_optimization()
