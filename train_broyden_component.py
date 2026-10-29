import numpy as np
import scipy.optimize as opt
import time

def objective_function(x):
    """
    Rosenbrock function.
    """
    return np.array([
        10 * (x[1] - x[0]**2),
        (1 - x[0])
    ])

def train():
    print("Starting optimization with Broyden's method (Good)...")
    start_time = time.time()

    x0 = np.array([-1.2, 1.0])

    result = opt.broyden1(objective_function, x0, f_tol=1e-5)

    end_time = time.time()

    print(f"Optimization finished in {end_time - start_time:.4f} seconds.")
    print(f"Final solution: {result}")

    final_f = objective_function(result)
    print(f"Final function value: {np.linalg.norm(final_f)}")

if __name__ == "__main__":
    train()
