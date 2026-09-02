import numpy as np
import os
import time

def spline_regression():
    np.random.seed(42)
    # Generate some non-linear data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.2, 100)

    # Let's create a natural cubic spline manually
    # We will choose some knots
    knots = np.array([2, 5, 8])

    # Create the design matrix X for cubic splines
    # Basis functions: 1, x, x^2, x^3, max(0, x - k)^3 for each knot k
    X = np.column_stack([np.ones_like(x), x, x**2, x**3])
    for k in knots:
        basis = np.maximum(0, x - k)**3
        X = np.column_stack([X, basis])

    # Solve for weights using ordinary least squares (X^T X)^-1 X^T y
    # Adding a small ridge penalty for stability
    lambda_reg = 1e-4
    XTX = X.T @ X + lambda_reg * np.eye(X.shape[1])
    XTy = X.T @ y
    weights = np.linalg.solve(XTX, XTy)

    # Predictions
    y_pred = X @ weights

    # Calculate MSE
    mse = np.mean((y - y_pred)**2)
    return mse

if __name__ == '__main__':
    print("Starting Spline Regression component training...")
    start_time = time.time()
    mse = spline_regression()
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    print(f"Final Mean Squared Error: {mse:.4f}")
    if mse < 0.1:
        print("Success: Spline Regression accurately modeled the non-linear data.")
    else:
        print("Failure: Spline Regression failed to model the data.")
