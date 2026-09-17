import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    """Rosenbrock function"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    """Gradient of the Rosenbrock function"""
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

def train_lbfgs():
    """
    Demonstrate Limited-memory BFGS (L-BFGS) optimization.
    L-BFGS approximates the BFGS algorithm using a limited amount of computer memory.
    It's particularly useful for optimization problems with many variables.
    Instead of storing the full dense nxn inverse Hessian approximation matrix,
    L-BFGS stores only a few vectors that represent the approximation implicitly.
    """
    print("Exploring L-BFGS (Limited-memory BFGS) Optimization Mathematically\n")

    # 1. Define objective function and initial point
    x0 = np.array([-1.2, 1.0])
    print(f"Objective Function: Rosenbrock Function")
    print(f"f(x, y) = (1 - x)^2 + 100(y - x^2)^2")
    print(f"Initial point: [x, y] = {x0}, Initial Loss: {rosenbrock(x0):.6f}\n")

    # 2. Run L-BFGS-B (Limited-memory BFGS with Bounds, though bounds aren't used here)
    # L-BFGS uses a two-loop recursion to compute the search direction
    # p_k = -H_k * \nabla f_k without explicitly forming H_k.
    print("Running L-BFGS optimization via SciPy...")
    res = minimize(
        rosenbrock,
        x0,
        method='L-BFGS-B',
        jac=rosenbrock_grad
    )

    # 3. Output results
    print(f"\nOptimization Completed.")
    print(f"Success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Number of Iterations: {res.nit}")
    print(f"Number of Function Evaluations: {res.nfev}")

    final_point = res.x
    final_loss = res.fun
    print(f"\nFinal Point (Optimal): [x, y] = [{final_point[0]:.6f}, {final_point[1]:.6f}]")
    print(f"Final Loss: {final_loss:.6f}")

    # Verify convergence to global minimum (1, 1)
    if np.allclose(final_point, [1.0, 1.0], atol=1e-4) and final_loss < 1e-8:
        print("\nSuccessfully converged to the global minimum (1, 1)!")
    else:
        print("\nFailed to converge to the global minimum.")

if __name__ == '__main__':
    train_lbfgs()
