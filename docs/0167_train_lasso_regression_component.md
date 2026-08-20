# Component: Lasso Regression

**Status:** Success

## Objective
To evaluate a Lasso Regression component mathematically in pure NumPy, testing its ability to perform linear regression with L1 regularization to prevent overfitting and encourage sparse weights.

## Mathematical Formulation
Lasso Regression finds the parameters $\theta = [\text{bias}, \text{weights}]$ that minimize the cost function:
$J(\theta) = \|X\theta - y\|^2 + \alpha \|\theta_{1:}\|_1$

This is solved iteratively using subgradient descent, since the L1 norm is not differentiable at 0.

## Implementation Details
- `LassoRegression`: Computes the solution using subgradient descent iteratively.
- The bias term is explicitly excluded from the regularization penalty.

## Verification
A script was successfully executed to train a Lasso Regression model on a synthetic dataset where one feature was irrelevant. The model successfully fitted the data, effectively pushing the weight of the irrelevant feature close to zero and achieved an MSE around 0.03.

train_lasso_regression_component
