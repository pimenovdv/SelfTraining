import numpy as np
import scipy.optimize

def rosenbrock(x):
    """The Rosenbrock function."""
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def main():
    print("Testing Basinhopping global optimization on the Rosenbrock function...")

    # Start from an initial guess
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # Basinhopping combines global stepping with local minimization
    # We use BFGS as the local minimizer (via minimizer_kwargs)
    minimizer_kwargs = {"method": "BFGS"}

    # Run the optimization
    result = scipy.optimize.basinhopping(
        rosenbrock,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=100,
        stepsize=0.5,
        T=1.0,
        disp=False
    )

    print("\nOptimization Finished!")
    print(f"Success: {result.lowest_optimization_result.success}")
    print(f"Message: {result.message}")
    print(f"Number of iterations: {result.nit}")
    print(f"Number of function evaluations: {result.nfev}")
    print(f"Best solution found: {result.x}")
    print(f"Function value at best solution: {result.fun}")

    # The global minimum of the Rosenbrock function is at (1, 1, ..., 1) with value 0
    expected_solution = np.ones_like(x0)
    expected_value = 0.0

    solution_error = np.linalg.norm(result.x - expected_solution)
    value_error = abs(result.fun - expected_value)

    print(f"\nSolution Error: {solution_error:.6f}")
    print(f"Value Error: {value_error:.6f}")

    if value_error < 1e-4:
        print("\nBasinhopping successfully found the global minimum!")
    else:
        print("\nBasinhopping failed to find the global minimum.")

if __name__ == "__main__":
    main()
