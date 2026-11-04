import torch
import torch.nn as nn
import numpy as np

def ellipsoid_method(f, grad_f, x0, radius=10.0, tol=1e-4, max_iter=1000):
    """
    Ellipsoid Method for Convex Optimization.
    Minimizes a convex function f(x) over a ball of given radius.

    Args:
        f (callable): Objective function.
        grad_f (callable): Gradient (or subgradient) of the objective function.
        x0 (torch.Tensor): Initial guess, inside the initial ellipsoid.
        radius (float): Radius of the initial spherical ellipsoid containing the minimizer.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

    Returns:
        x (torch.Tensor): Estimated minimum.
        history (list): Objective values over iterations.
    """
    x = x0.clone().detach().double()
    n = x.shape[0]

    # Initialize the ellipsoid matrix P = R^2 * I
    P = (radius ** 2) * torch.eye(n, dtype=torch.float64)

    history = []

    for i in range(max_iter):
        # Calculate subgradient at current center
        g = grad_f(x).double()

        # Check convergence based on the norm of the gradient
        g_norm = torch.norm(g)
        if g_norm < tol:
            break

        history.append(f(x).item())

        # Normalize gradient
        g_normed = g / g_norm

        # Update center x
        step = (1 / (n + 1)) * torch.matmul(P, g_normed) / torch.sqrt(torch.dot(g_normed, torch.matmul(P, g_normed)))
        x = x - step

        # Update ellipsoid matrix P
        Pg = torch.matmul(P, g_normed)
        gPg = torch.dot(g_normed, Pg)
        P = (n ** 2 / (n ** 2 - 1)) * (P - (2 / (n + 1)) * torch.ger(Pg, Pg) / gPg)

    return x.float(), history

if __name__ == '__main__':
    # Test with a simple strongly convex quadratic function
    # f(x) = x_1^2 + 2*x_2^2 + ... + n*x_n^2
    n_dim = 5
    A = torch.diag(torch.arange(1, n_dim + 1, dtype=torch.float64))

    def f(x):
        return 0.5 * torch.dot(x, torch.matmul(A, x))

    def grad_f(x):
        return torch.matmul(A, x)

    x0 = torch.ones(n_dim) * 2.0

    print("Testing Ellipsoid Method...")
    x_opt, history = ellipsoid_method(f, grad_f, x0, radius=20.0, max_iter=200)

    print(f"Optimal x: {x_opt}")
    print(f"Final loss: {f(x_opt.double()).item():.6f}")
    assert f(x_opt.double()).item() < 1.0, "Ellipsoid Method failed to converge!"
    print("Test passed.")
