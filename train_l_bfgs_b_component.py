import numpy as np
from scipy.optimize import minimize
import time

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    df_dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    df_dx1 = 200 * (x[1] - x[0]**2)
    return np.array([df_dx0, df_dx1])

def l_bfgs_b_optimization():
    x0 = np.array([-1.2, 1.0])
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]

    start_time = time.time()

    res = minimize(
        fun=rosenbrock,
        x0=x0,
        jac=rosenbrock_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxcor': 10, 'ftol': 1e-9, 'gtol': 1e-5}
    )

    execution_time = time.time() - start_time

    print(f"L-BFGS-B optimization completed in {execution_time:.4f} seconds.")
    print(f"Optimal parameters: {res.x}")
    print(f"Final Loss: {res.fun:.6f}")

    return res.x, res.fun

if __name__ == "__main__":
    l_bfgs_b_optimization()
