import numpy as np

def rosenbrock(x):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    return np.array([
        -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2),
        200.0 * (x[1] - x[0]**2)
    ])

def rosenbrock_hess(x):
    return np.array([
        [2.0 - 400.0 * x[1] + 1200.0 * x[0]**2, -400.0 * x[0]],
        [-400.0 * x[0], 200.0]
    ])

def trcg(obj_func, grad_func, hess_func, x0, radius=1.0, max_iter=100, tol=1e-6):
    x = np.array(x0, dtype=float)

    for i in range(max_iter):
        g = grad_func(x)
        if np.linalg.norm(g) < tol:
            print(f"Converged at iteration {i}")
            break

        H = hess_func(x)

        # Steihaug's Conjugate Gradient for Trust Region Subproblem
        z = np.zeros_like(x)
        r = g.copy()
        d = -r.copy()

        if np.linalg.norm(r) < 1e-10:
            p = z
        else:
            for j in range(len(x)):
                Bd = np.dot(H, d)
                dBd = np.dot(d, Bd)

                if dBd <= 0:
                    # Negative curvature, move to boundary
                    a_quad = np.dot(d, d)
                    b_quad = 2 * np.dot(z, d)
                    c_quad = np.dot(z, z) - radius**2
                    tau = (-b_quad + np.sqrt(b_quad**2 - 4 * a_quad * c_quad)) / (2 * a_quad)
                    p = z + tau * d
                    break

                alpha = np.dot(r, r) / dBd
                z_next = z + alpha * d
                if np.linalg.norm(z_next) >= radius:
                    # Intersects boundary
                    a_quad = np.dot(d, d)
                    b_quad = 2 * np.dot(z, d)
                    c_quad = np.dot(z, z) - radius**2
                    tau = (-b_quad + np.sqrt(b_quad**2 - 4 * a_quad * c_quad)) / (2 * a_quad)
                    p = z + tau * d
                    break

                r_next = r + alpha * Bd
                if np.linalg.norm(r_next) < 1e-10:
                    p = z_next
                    break

                beta = np.dot(r_next, r_next) / np.dot(r, r)
                d = -r_next + beta * d

                z = z_next
                r = r_next
            else:
                p = z

        # Evaluate reduction
        actual_reduction = obj_func(x) - obj_func(x + p)
        predicted_reduction = -np.dot(g, p) - 0.5 * np.dot(p, np.dot(H, p))

        if predicted_reduction == 0:
            rho = 0
        else:
            rho = actual_reduction / predicted_reduction

        if rho > 0.75:
            radius = min(2 * radius, 100.0)
        elif rho < 0.25:
            radius = 0.25 * radius

        if rho > 0:
            x = x + p

    return x

if __name__ == "__main__":
    print("Testing Trust Region Conjugate Gradient (TRCG) on Rosenbrock function...")
    x_opt = trcg(rosenbrock, rosenbrock_grad, rosenbrock_hess, x0=[-1.2, 1.0])
    print(f"Optimal parameters: {x_opt}")
    print(f"Minimum value: {rosenbrock(x_opt)}")
    print("TRCG optimization successful!")
