import numpy as np

def levenberg_marquardt(x, y, func, jacobian, initial_params, max_iters=100, tol=1e-6, lambda_init=0.01):
    """
    Levenberg-Marquardt algorithm for non-linear least squares.
    """
    params = np.array(initial_params, dtype=float)
    lam = lambda_init

    for i in range(max_iters):
        residuals = y - func(x, *params)
        mse = np.mean(residuals**2)

        J = jacobian(x, *params)

        # Levenberg-Marquardt step
        H = J.T @ J
        H_lm = H + lam * np.diag(np.diag(H))

        g = J.T @ residuals

        try:
            delta = np.linalg.solve(H_lm, g)
        except np.linalg.LinAlgError:
            break

        new_params = params + delta
        new_residuals = y - func(x, *new_params)
        new_mse = np.mean(new_residuals**2)

        if new_mse < mse:
            # Step accepted, decrease lambda
            params = new_params
            lam /= 10
            if abs(mse - new_mse) < tol:
                break
        else:
            # Step rejected, increase lambda
            lam *= 10

    return params, mse

if __name__ == "__main__":
    print("Testing Levenberg-Marquardt Component...")

    # Model: a * exp(b * x)
    def model_func(x, a, b):
        return a * np.exp(b * x)

    def jacobian_func(x, a, b):
        J = np.zeros((len(x), 2))
        J[:, 0] = np.exp(b * x)
        J[:, 1] = a * x * np.exp(b * x)
        return J

    # Generate true data
    np.random.seed(42)
    x_data = np.linspace(0, 4, 50)
    true_a, true_b = 2.5, 1.3
    y_true = model_func(x_data, true_a, true_b)
    y_noisy = y_true + 0.1 * np.random.randn(len(x_data))

    initial_guess = [1.0, 1.0]

    params_opt, final_mse = levenberg_marquardt(
        x_data, y_noisy, model_func, jacobian_func, initial_guess
    )

    print(f"True parameters: a={true_a}, b={true_b}")
    print(f"Optimized parameters: a={params_opt[0]:.4f}, b={params_opt[1]:.4f}")
    print(f"Final MSE: {final_mse:.6f}")

    if final_mse < 0.1:
        print("Levenberg-Marquardt optimization successful.")
    else:
        print("Levenberg-Marquardt optimization failed.")
        exit(1)
