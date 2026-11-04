import numpy as np

def nonlinear_conjugate_gradient(f, grad_f, x0, num_iters=100, tol=1e-5):
    """
    Minimizes f(x) using the Non-linear Conjugate Gradient Method (Fletcher-Reeves).
    """
    x = np.copy(x0)
    g = grad_f(x)
    d = -g

    for _ in range(num_iters):
        if np.linalg.norm(g) < tol:
            break

        # Line search (simple backtracking)
        alpha = 1.0
        c = 1e-4
        rho = 0.5
        while f(x + alpha * d) > f(x) + c * alpha * np.dot(g, d):
            alpha *= rho

        x_new = x + alpha * d
        g_new = grad_f(x_new)

        # Fletcher-Reeves beta update
        beta = np.dot(g_new, g_new) / (np.dot(g, g) + 1e-10)

        d = -g_new + beta * d
        x = x_new
        g = g_new

    return x

def test_nonlinear_cg():
    # Min f(x, y) = 10*(y - x^2)^2 + (1 - x)^2  (Rosenbrock)
    def f(x):
        return 10 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad_f(x):
        df_dx0 = -40 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        df_dx1 = 20 * (x[1] - x[0]**2)
        return np.array([df_dx0, df_dx1])

    x0 = np.array([-1.2, 1.0])

    x_opt = nonlinear_conjugate_gradient(f, grad_f, x0, num_iters=1000)

    print("Optimal x:", x_opt)
    assert np.allclose(x_opt, [1.0, 1.0], atol=1e-2)
    print("Non-linear Conjugate Gradient component tested successfully.")

if __name__ == '__main__':
    test_nonlinear_cg()
