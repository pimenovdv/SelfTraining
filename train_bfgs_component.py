import numpy as np
import time

def rosenbrock(x):
    """Rosenbrock function"""
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosenbrock_grad(x):
    """Gradient of the Rosenbrock function"""
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm - xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1 - xm)
    der[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
    der[-1] = 200*(x[-1] - x[-2]**2)
    return der

def quadratic(x, A, b):
    return 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)

def quadratic_grad(x, A, b):
    return np.dot(A, x) - b

def backtracking_line_search(f, grad_f, x, p, grad_x, alpha=1.0, c=1e-4, rho=0.9):
    """Backtracking line search for Armijo condition."""
    f_x = f(x)
    while f(x + alpha * p) > f_x + c * alpha * np.dot(grad_x, p):
        alpha *= rho
        if alpha < 1e-10:
            break
    return alpha

def bfgs_minimize(f, grad_f, x0, tol=1e-5, max_iter=1000):
    n = len(x0)
    H = np.eye(n)
    x = x0.copy()
    grad_x = grad_f(x)

    for i in range(max_iter):
        if np.linalg.norm(grad_x) < tol:
            print(f"Converged in {i} iterations.")
            break

        p = -np.dot(H, grad_x)

        alpha = backtracking_line_search(f, grad_f, x, p, grad_x)

        x_new = x + alpha * p
        grad_new = grad_f(x_new)

        s = x_new - x
        y = grad_new - grad_x

        ys = np.dot(y, s)
        if ys > 1e-10:
            rho = 1.0 / ys
            I = np.eye(n)
            H = np.dot((I - rho * np.outer(s, y)), np.dot(H, (I - rho * np.outer(y, s)))) + rho * np.outer(s, s)

        x = x_new
        grad_x = grad_new

    return x, f(x)

if __name__ == "__main__":
    print("Initializing BFGS Optimization Component...")

    # 1. Quadratic function test
    print("\n--- Test 1: Quadratic Function ---")
    np.random.seed(42)
    n_dim = 10
    A = np.random.randn(n_dim, n_dim)
    A = np.dot(A.T, A) # make symmetric positive definite
    b = np.random.randn(n_dim)

    f = lambda x: quadratic(x, A, b)
    grad_f = lambda x: quadratic_grad(x, A, b)

    x0 = np.zeros(n_dim)

    start_time = time.time()
    x_opt, f_opt = bfgs_minimize(f, grad_f, x0)
    end_time = time.time()

    print(f"Optimal value: {f_opt:.6f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # Check against true solution
    true_x = np.linalg.solve(A, b)
    print(f"Error norm: {np.linalg.norm(x_opt - true_x):.6e}")

    # 2. Rosenbrock function test
    print("\n--- Test 2: Rosenbrock Function ---")
    x0_rosen = np.array([-1.2, 1.0, -1.2, 1.0])

    start_time = time.time()
    x_opt_rosen, f_opt_rosen = bfgs_minimize(rosenbrock, rosenbrock_grad, x0_rosen)
    end_time = time.time()

    print(f"Optimal value: {f_opt_rosen:.6f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Optimal point: {x_opt_rosen}")
