import torch
import torch.nn as nn
import torch.optim as optim
import time

class AugmentedLagrangianModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 2D problem
        self.x = nn.Parameter(torch.tensor([2.0, 2.0]))

    def forward(self):
        return self.x

def objective_function(x):
    # f(x,y) = x^2 + y^2
    return x[0]**2 + x[1]**2

def equality_constraint(x):
    # g(x,y) = x + y - 1 = 0
    return x[0] + x[1] - 1.0

def train_augmented_lagrangian():
    model = AugmentedLagrangianModel()

    # Dual variables (Lagrange multipliers)
    lambda_mult = torch.tensor(0.0)

    # Penalty parameter
    rho = 1.0

    # Optimizer for the primal variables
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_outer_iterations = 50
    num_inner_iterations = 100

    start_time = time.time()
    for outer in range(num_outer_iterations):
        # Minimize augmented Lagrangian w.r.t. primal variables
        for inner in range(num_inner_iterations):
            optimizer.zero_grad()

            x = model()
            f_val = objective_function(x)
            g_val = equality_constraint(x)

            # L(x, lambda, rho) = f(x) + lambda * g(x) + (rho / 2) * g(x)^2
            lagrangian = f_val + lambda_mult * g_val + (rho / 2.0) * (g_val ** 2)

            lagrangian.backward()
            optimizer.step()

        # Update dual variables and penalty parameter
        with torch.no_grad():
            x_opt = model()
            g_val_opt = equality_constraint(x_opt)

            lambda_mult += rho * g_val_opt
            rho *= 1.1 # Increase penalty slightly

    end_time = time.time()

    x_final = model().detach()
    f_final = objective_function(x_final).item()
    g_final = equality_constraint(x_final).item()

    print(f"Optimization completed in {end_time - start_time:.4f} seconds.")
    print(f"Final solution: x = {x_final.numpy()}")
    print(f"Objective value: {f_final:.4f}")
    print(f"Constraint violation: {g_final:.4f}")

    # Theoretical minimum for min x^2+y^2 s.t. x+y=1 is x=0.5, y=0.5, min=0.5
    assert torch.allclose(x_final, torch.tensor([0.5, 0.5]), atol=1e-2)
    assert abs(g_final) < 1e-2

if __name__ == "__main__":
    train_augmented_lagrangian()
