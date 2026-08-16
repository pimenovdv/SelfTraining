# Gradient Boosting Component

## Overview
This component implements Gradient Boosting Regression from scratch using pure NumPy. Gradient Boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

## Mathematical Foundation
The gradient boosting algorithm involves three elements:
1.  **A loss function to be optimized.** (e.g., Mean Squared Error for regression).
2.  **A weak learner to make predictions.** (e.g., Decision Stumps or shallow trees).
3.  **An additive model to add weak learners to minimize the loss function.**

For regression with Mean Squared Error (MSE), the negative gradient of the loss function is simply the residual (difference between true value and predicted value).
Thus, in each iteration, a new weak learner is fitted to the residuals of the current ensemble.

Let $F_m(x)$ be the model at stage $m$.
$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$
where $h_m(x)$ is the weak learner fitted to the pseudo-residuals $r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x)=F_{m-1}(x)}$ and $\nu$ is the learning rate.

## Implementation Details
- `DecisionStump`: A simple decision tree with a depth of 1, acting as the weak learner. It searches for the best feature and threshold to minimize the MSE on the target values (residuals in the boosting context).
- `GradientBoostingRegressor`: The main estimator. It initializes predictions with the mean of the target. In each iteration, it computes residuals, fits a `DecisionStump` to the residuals, and updates the predictions scaled by a `learning_rate`.

## Verification
A script was successfully executed to train a Gradient Boosting Regressor on a noisy sine wave dataset. The model successfully fitted the data and achieved an MSE below 0.1, demonstrating its capability to capture non-linear relationships.

train_gradient_boosting_component
