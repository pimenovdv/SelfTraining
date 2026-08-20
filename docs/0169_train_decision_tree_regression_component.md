# Experiment 0169: Decision Tree Regression Component

**Hypothesis:** By recursively splitting the feature space based on threshold values that maximize variance reduction (minimizing Mean Squared Error within splits), a Decision Tree can effectively learn non-linear regression functions without requiring feature scaling or distributional assumptions.

**Action:**
- Implemented a Decision Tree Regressor mathematically in pure NumPy.
- The model recursively finds splits that maximize the difference between the variance of the parent node and the weighted variance of the child nodes.
- Tested the implementation on a synthetic noisy sine wave dataset.

**Outcome:**
- The model successfully fit the non-linear dataset.
- Achieved a low Mean Squared Error (MSE), indicating that the recursive splitting logic and leaf value calculation (mean of target values) correctly approximated the underlying function.

**Next Steps:**
- Extend the model to an ensemble by implementing a Random Forest Regressor to improve generalization and reduce variance.

**Script:** `train_decision_tree_regression_component.py`
**Status:** Success
