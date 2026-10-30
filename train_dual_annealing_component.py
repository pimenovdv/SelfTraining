import numpy as np
from scipy.optimize import dual_annealing

def rosenbrock(x):
    """Rosenbrock function."""
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

if __name__ == "__main__":
    print("Starting Dual Annealing training...")
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    # Run dual annealing
    result = dual_annealing(rosenbrock, bounds, seed=42)

    print(f"Final Best Point: {result.x}")
    print(f"Final Best Eval: {result.fun:.6e}")
    print(f"Message: {result.message}")

    assert result.fun < 1e-4, "Dual Annealing failed to converge near the global minimum."
    print("Training complete and successful.")
