import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    """Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

if __name__ == "__main__":
    print("Testing Powell's Method on the Rosenbrock function...")
    x0 = np.array([-1.2, 1.0])

    res = minimize(rosenbrock, x0, method='Powell', options={'xtol': 1e-8, 'disp': True})

    print("\nInitial position:", x0)
    print("Final position:", res.x)
    print("Final value:", res.fun)
    print("Number of iterations:", res.nit)

    assert res.fun < 1e-4, "Powell's Method failed to converge on the Rosenbrock function"
    print("Powell's Method successfully converged!")
