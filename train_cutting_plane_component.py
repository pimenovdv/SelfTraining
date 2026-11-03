import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

# A basic implementation of Kelley's Cutting Plane Method for convex optimization

class ConvexFunction:
    # Example convex function to minimize: f(x) = x_1^2 + x_2^2
    def __call__(self, x):
        return x[0]**2 + x[1]**2

    def gradient(self, x):
        return torch.tensor([2.0 * x[0], 2.0 * x[1]])

def train_cutting_plane():
    start_time = time.time()

    func = ConvexFunction()

    # We want to minimize f(x). We reformulate as:
    # min t
    # s.t. f(x) <= t

    # In Kelley's method, we approximate f(x) by a set of linear lower bounds.
    # f(x) >= f(x_k) + g_k^T (x - x_k)
    # So we want to solve:
    # min t
    # s.t. f(x_k) + g_k^T (x - x_k) <= t for all k

    # Since we can't easily solve this unbounded linear program without a bounding box,
    # we add simple bounds -5 <= x_1, x_2 <= 5

    # Initial guess
    x_k = torch.tensor([4.0, 4.0])

    # Store cutting planes: list of (f_val, grad, x_point)
    planes = []

    num_iterations = 20

    # For solving the LP subproblem, we'll use a simple PyTorch gradient descent on a penalty method
    # Since we don't have scipy.optimize.linprog readily available and standard PyTorch
    # doesn't have an LP solver, we solve the constrained subproblem via Adam optimization

    for i in range(num_iterations):
        # Evaluate function and gradient at x_k
        f_val = func(x_k)
        grad = func.gradient(x_k)

        planes.append((f_val.item(), grad.clone(), x_k.clone()))

        # Solve subproblem: min t s.t. f(x_j) + g_j^T (x - x_j) <= t
        # We define a neural network parameter for x and t
        x_var = nn.Parameter(torch.tensor([0.0, 0.0])) # Start at origin for subproblem
        t_var = nn.Parameter(torch.tensor(0.0))

        optimizer = optim.Adam([x_var, t_var], lr=0.1)

        for _ in range(1000): # inner optimization
            optimizer.zero_grad()

            # Objective
            loss = t_var

            # Cutting plane constraints
            penalty = 0.0
            for fj, gj, xj in planes:
                constraint_val = fj + torch.dot(gj, (x_var - xj)) - t_var
                penalty += torch.clamp(constraint_val, min=0.0)**2 * 1000.0 # quadratic penalty

            # Box constraints
            penalty += torch.clamp(x_var[0] - 5.0, min=0.0)**2 * 1000.0
            penalty += torch.clamp(-5.0 - x_var[0], min=0.0)**2 * 1000.0
            penalty += torch.clamp(x_var[1] - 5.0, min=0.0)**2 * 1000.0
            penalty += torch.clamp(-5.0 - x_var[1], min=0.0)**2 * 1000.0

            total_loss = loss + penalty
            total_loss.backward()
            optimizer.step()

        x_k = x_var.detach()

    end_time = time.time()

    print(f"Optimization completed in {end_time - start_time:.4f} seconds.")
    print(f"Final solution: x = {x_k.numpy()}")
    print(f"Objective value: {func(x_k):.4f}")

    # Theoretical minimum for x^2 + y^2 is (0,0)
    assert torch.allclose(x_k, torch.tensor([0.0, 0.0]), atol=0.2)

if __name__ == "__main__":
    train_cutting_plane()
