# Experiment 0170: Random Forest Regression Component

**Hypothesis:** By training an ensemble of Decision Trees on bootstrap samples of the dataset and selecting random feature subsets for each split, a Random Forest Regressor can significantly reduce the variance and overfitting typically associated with individual decision trees, leading to lower Mean Squared Error on non-linear regression tasks on unseen data.

**Action:**
- Implemented a Random Forest Regressor mathematically in pure NumPy.
- Constructed an ensemble of Decision Tree Regressors using bagging (bootstrap aggregation) and random feature selection.
- Used the mean of predictions from all individual trees for the final output.
- Tested generalization on a held-out test set from a noisy sine wave distribution and compared with a single Decision Tree.

**Outcome:**
- The implementation successfully fit the non-linear dataset while reducing overfitting.
- Random Forest Test MSE (0.0637) was lower than a single Decision Tree Test MSE (0.0895), verifying the reduction in variance and improved generalization.
- Status: Success

**Next Steps:**
- Explore advanced boosting regression models such as XGBoost or LightGBM equivalents mathematically.

**Script:** `train_random_forest_regression_component.py`
