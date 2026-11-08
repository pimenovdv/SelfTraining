import numpy as np

def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosenbrock_grad(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2]**2)
    return der

def line_search(f, grad_f, x, p, alpha=1.0, c1=1e-4, rho=0.5):
    while f(x + alpha * p) > f(x) + c1 * alpha * np.dot(grad_f(x), p):
        alpha *= rho
    return alpha

def non_linear_cg(f, grad_f, x0, tol=1e-5, max_iter=1000):
    x = x0.copy()
    g = grad_f(x)
    p = -g
    k = 0

    while np.linalg.norm(g) > tol and k < max_iter:
        alpha = line_search(f, grad_f, x, p)
        x_new = x + alpha * p
        g_new = grad_f(x_new)

        # Fletcher-Reeves beta
        beta = np.dot(g_new, g_new) / np.dot(g, g)

        p = -g_new + beta * p
        g = g_new
        x = x_new
        k += 1

    return x, f(x), k

if __name__ == '__main__':
    print("Testing Non-linear Conjugate Gradient on Rosenbrock function...")
    x0 = np.array([-1.2, 1.0])
    x_opt, f_opt, iters = non_linear_cg(rosenbrock, rosenbrock_grad, x0)
    print(f"Optimal x: {x_opt}")
    print(f"Minimum value: {f_opt}")
    print(f"Iterations: {iters}")
    assert np.allclose(x_opt, [1.0, 1.0], atol=1e-2), f"Optimization failed: {x_opt}"
    print("Non-linear Conjugate Gradient completed successfully.")
