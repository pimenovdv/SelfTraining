# Component: Ridge Regression

**Status:** Success

## Objective
To evaluate a Ridge Regression component mathematically in pure NumPy, testing its ability to perform linear regression with L2 regularization to prevent overfitting and handle multicollinearity.

## Mathematical Formulation
Ridge Regression finds the parameters $\theta = [\text{bias}, \text{weights}]$ that minimize the cost function:
$J(\theta) = \|X\theta - y\|^2 + \alpha \|\theta_{1:}\|^2$

The closed-form solution is:
$\theta = (X^T X + \alpha I)^{-1} X^T y$

where $X$ is the design matrix (with a column of ones for the bias), $y$ is the target vector, $I$ is the identity matrix (with the first element set to 0 to avoid regularizing the bias), and $\alpha$ is the regularization strength.

## Implementation Details
- `RidgeRegression`: Computes the closed-form solution using NumPy's linear algebra module (`np.linalg.inv`).
- The bias term is explicitly excluded from the regularization penalty by modifying the identity matrix.

## Verification
A script was successfully executed to train a Ridge Regression model on a synthetic dataset. The model successfully fitted the data and achieved an MSE around 0.01, demonstrating its capability to recover true weights and bias under noise and regularization.

train_ridge_regression_component
