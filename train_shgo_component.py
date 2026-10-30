import numpy as np
import time
from scipy.optimize import shgo, rosen

def shgo_optimization():
    # Define bounds for the Rosenbrock function
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]

    start_time = time.time()

    # SHGO optimization
    res = shgo(rosen, bounds)

    end_time = time.time()
    execution_time = end_time - start_time

    print("SHGO optimization results:")
    print(f"Optimal parameters: {res.x}")
    print(f"Optimal function value (loss): {res.fun:.6f}")
    print(f"Execution time: {execution_time:.4f} seconds")

    # Verify convergence to the global minimum (1, 1, 1, 1, 1)
    expected_optimum = np.ones(5)
    np.testing.assert_allclose(res.x, expected_optimum, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res.fun, 0.0, atol=1e-5)

    print("Optimization successful and verified.")

if __name__ == "__main__":
    shgo_optimization()
