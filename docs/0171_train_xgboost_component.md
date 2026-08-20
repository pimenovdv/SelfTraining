# XGBoost Regression Component

## Objective
Implement and evaluate an XGBoost Regressor mathematically in pure NumPy to demonstrate advanced gradient boosting with second-order Taylor expansion approximations (gradients and hessians) and L2 regularization.

## Implementation Details
The component builds an ensemble of regression trees where each split is chosen by maximizing a gain function defined by the gradients and hessians of the loss function. The predictions are scaled by a learning rate.

## Results
The component was executed successfully on a noisy sine wave dataset.
- MSE: 0.0041
- The model effectively fit the non-linear relationship.

**Script:** `train_xgboost_component.py`
**Status:** Success
