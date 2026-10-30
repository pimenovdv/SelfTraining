import torch
import torch.nn as nn
import torch.optim as optim
import time
from scipy.optimize import minimize
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train_cobyla():
    print("Starting COBYLA optimization...")
    start_time = time.time()

    # Simple quadratic function: f(x) = x[0]^2 + x[1]^2
    # Global minimum is at (0, 0)

    def objective(x):
        return x[0]**2 + x[1]**2

    # Constraint: x[0] + x[1] >= 1  => x[0] + x[1] - 1 >= 0
    def constraint1(x):
        return x[0] + x[1] - 1

    # Initial guess
    x0 = np.array([2.0, 2.0])
    print(f"Initial guess: {x0}, Initial value: {objective(x0):.6f}")

    # Constraints for COBYLA
    cons = [{'type': 'ineq', 'fun': constraint1}]

    # Run optimization
    res = minimize(objective, x0, method='COBYLA', constraints=cons, options={'disp': True, 'maxiter': 500, 'rhobeg': 1.0})

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Optimization finished in {execution_time:.2f} seconds")
    print(f"Success: {res.success}")
    print(f"Status message: {res.message}")
    if res.success:
        print(f"Final parameters: {res.x}")
        print(f"Final function value: {res.fun:.6f}")
        print(f"Number of function evaluations: {res.nfev}")

    if res.success:
        print("COBYLA optimization successful.")
    else:
        print("COBYLA optimization failed to converge.")

if __name__ == "__main__":
    train_cobyla()
