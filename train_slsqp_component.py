import numpy as np
from scipy.optimize import minimize
import time

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosen_der(x):
    """Derivative of the Rosenbrock function."""
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm - xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1 - xm)
    der[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
    der[-1] = 200*(x[-1] - x[-2]**2)
    return der

def run_slsqp_optimization():
    # Initial guess
    x0 = np.array([-1.2, 1.0])

    # Define some constraints for SLSQP
    # Example constraint: x[0] + 2*x[1] <= 1 => 1 - x[0] - 2*x[1] >= 0
    cons = ({'type': 'ineq', 'fun': lambda x: 1 - x[0] - 2 * x[1]})

    # Bounds for x and y
    bnds = ((-1.5, 1.5), (-1.5, 1.5))

    print("Starting SLSQP Optimization on Rosenbrock function with constraints...")
    start_time = time.time()

    res = minimize(rosen, x0, method='SLSQP', jac=rosen_der, bounds=bnds, constraints=cons, options={'disp': True})

    end_time = time.time()

    print(f"Optimization finished in {end_time - start_time:.4f} seconds.")
    print(f"Success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Final parameters: {res.x}")
    print(f"Final loss: {res.fun:.6f}")

    if res.success:
        print("SLSQP successfully found a constrained minimum.")
    else:
        print("SLSQP failed to converge.")

if __name__ == '__main__':
    run_slsqp_optimization()
