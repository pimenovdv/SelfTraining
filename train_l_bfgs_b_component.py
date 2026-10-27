import numpy as np
from scipy.optimize import minimize

def l_bfgs_b_optimize(func, jacobian, initial_guess, bounds, max_iter=100):
    """
    Optimizes a function using the L-BFGS-B algorithm.
    """
    result = minimize(
        func,
        initial_guess,
        jac=jacobian,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter}
    )
    return result.x, result.fun

def test_l_bfgs_b_component():
    print("Testing L-BFGS-B Optimization Component...")

    # Objective function: f(x, y) = (x - 2)^2 + (y - 3)^2
    def obj_func(params):
        x, y = params
        return (x - 2)**2 + (y - 3)**2

    def obj_jac(params):
        x, y = params
        return np.array([2 * (x - 2), 2 * (y - 3)])

    initial_guess = np.array([0.0, 0.0])
    bounds = [(-5, 1.5), (-5, 2.5)] # x <= 1.5, y <= 2.5

    optimized_params, final_val = l_bfgs_b_optimize(obj_func, obj_jac, initial_guess, bounds)

    print(f"Optimized parameters: {optimized_params}")
    print(f"Final value: {final_val:.6f}")

    # Expected solution due to bounds: x = 1.5, y = 2.5
    expected_params = np.array([1.5, 2.5])

    assert np.allclose(optimized_params, expected_params, atol=1e-4), "L-BFGS-B optimization failed to reach expected constrained minimum."
    print("Test passed successfully.")

if __name__ == "__main__":
    test_l_bfgs_b_component()
