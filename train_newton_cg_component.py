import numpy as np
import time

def rosenbrock(x):
    """The Rosenbrock function."""
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosenbrock_grad(x):
    """The gradient of the Rosenbrock function."""
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm - xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1 - xm)
    der[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
    der[-1] = 200*(x[-1] - x[-2]**2)
    return der

def rosenbrock_hess(x):
    """The Hessian matrix of the Rosenbrock function."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i] += 1200 * x[i]**2 - 400 * x[i+1] + 2
        H[i, i+1] += -400 * x[i]
        H[i+1, i] += -400 * x[i]
        H[i+1, i+1] += 200
    return H

def cg(A, b, x0=None, tol=1e-5, max_iter=None):
    """Conjugate Gradient method to solve Ax = b."""
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    r = b - A @ x
    p = r.copy()
    rsold = np.dot(r, r)

    if max_iter is None:
        max_iter = len(b) * 2

    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (np.dot(p, Ap) + 1e-10)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / (rsold + 1e-10)) * p
        rsold = rsnew
    return x

def newton_cg(obj_func, grad_func, hess_func, x0, tol=1e-5, max_iter=100):
    """Newton-CG (Truncated Newton) Optimization."""
    x = np.array(x0, dtype=float)
    history = [obj_func(x)]

    for i in range(max_iter):
        g = grad_func(x)
        if np.linalg.norm(g) < tol:
            break

        H = hess_func(x)

        # Add small regularization to make sure H is positive definite
        eigvals = np.linalg.eigvals(H)
        min_eig = np.min(eigvals)
        if min_eig <= 0:
            H = H + (-min_eig + 1e-5) * np.eye(len(x))

        # Use CG to solve H p = -g
        p = cg(H, -g)

        # Line search (simple backtracking)
        alpha = 1.0
        c = 1e-4
        rho = 0.5
        obj_val = obj_func(x)

        while obj_func(x + alpha * p) > obj_val + c * alpha * np.dot(g, p):
            alpha *= rho
            if alpha < 1e-10:
                break

        x = x + alpha * p
        history.append(obj_func(x))

    return x, history

def main():
    print("Testing Newton-CG Optimization Component")
    np.random.seed(42)

    x0 = np.array([-1.2, 1.0])

    start_time = time.time()
    x_opt, history = newton_cg(rosenbrock, rosenbrock_grad, rosenbrock_hess, x0)
    end_time = time.time()

    print(f"Initial point: {x0}")
    print(f"Optimal point: {x_opt}")
    print(f"Optimal value: {history[-1]:.6f}")
    print(f"Number of iterations: {len(history)-1}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()
