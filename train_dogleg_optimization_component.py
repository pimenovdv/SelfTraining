import time
import numpy as np
from scipy.optimize import minimize
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rosenbrock(x):
    """The Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosenbrock_der(x):
    """The derivative of the Rosenbrock function."""
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2]**2)
    return der

def rosenbrock_hess(x):
    """The Hessian matrix of the Rosenbrock function."""
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H


def train_dogleg():
    logging.info("Starting Dogleg optimization on the Rosenbrock function.")
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    start_time = time.time()

    res = minimize(rosenbrock, x0, method='dogleg', jac=rosenbrock_der, hess=rosenbrock_hess, options={'disp': True})

    end_time = time.time()
    execution_time = end_time - start_time

    logging.info(f"Execution time: {execution_time:.4f} seconds")
    logging.info(f"Final parameters: {res.x}")
    logging.info(f"Final loss: {res.fun:.6f}")
    logging.info(f"Success: {res.success}")

    if not res.success:
        logging.error(f"Optimization failed: {res.message}")

if __name__ == "__main__":
    train_dogleg()
