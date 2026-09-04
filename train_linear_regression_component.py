"""
Mathematical implementation of Linear Regression using the Normal Equation.
"""
import numpy as np

def train_linear_regression(X, y):
    # Add bias term (column of 1s)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Normal Equation: theta = (X^T * X)^-1 * X^T * y
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    return theta_best

def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(theta)

if __name__ == "__main__":
    print("Testing Linear Regression using Normal Equation...")

    # Generate synthetic linear data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    theta_best = train_linear_regression(X, y)

    print("Expected theta (approx): intercept=4, slope=3")
    print(f"Calculated theta:\n{theta_best}")

    # Prediction
    X_new = np.array([[0], [2]])
    y_predict = predict(X_new, theta_best)
    print(f"Predictions for X=[[0], [2]]:\n{y_predict}")

    mse = np.mean((predict(X, theta_best) - y) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")
    assert mse < 1.5, "MSE is too high, model failed to learn."
    print("Linear Regression component mathematically verified successfully.")
