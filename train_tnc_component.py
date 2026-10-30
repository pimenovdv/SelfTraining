import numpy as np
from scipy.optimize import minimize
import time

def rosenbrock(x):
    """The Rosenbrock function."""
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosenbrock_der(x):
    """The derivative of the Rosenbrock function."""
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm - xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1 - xm)
    der[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
    der[-1] = 200*(x[-1] - x[-2]**2)
    return der

def test_tnc_optimizer():
    """Tests the TNC optimizer on the Rosenbrock function."""
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    print("Running TNC Optimizer on Rosenbrock function...")
    start_time = time.time()

    # Truncated Newton (TNC) algorithm
    res = minimize(rosenbrock, x0, method='TNC', jac=rosenbrock_der, options={'maxfun': 500, 'ftol': 1e-8})

    end_time = time.time()
    duration = end_time - start_time

    print(f"Optimization finished in {duration:.4f} seconds.")
    print(f"Success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Number of iterations: {res.nit}")
    print(f"Final parameters: {res.x}")
    print(f"Final loss (f(x)): {res.fun}")

    assert res.success, f"TNC optimizer failed to converge: {res.message}"
    assert np.allclose(res.x, np.ones_like(res.x), atol=1e-3), "Solution is not close to the global minimum [1, 1, 1, 1, 1]"

if __name__ == "__main__":
    test_tnc_optimizer()
