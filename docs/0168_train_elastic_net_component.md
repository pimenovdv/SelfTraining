# Experiment 0168: Elastic Net Regression Component

**Hypothesis:** By combining L1 and L2 regularization penalties, an Elastic Net Regression model can effectively perform feature selection like Lasso while maintaining the regularization properties of Ridge regression, preventing overfitting in datasets with highly correlated features.

**Action:**
- Implemented an Elastic Net Regression model mathematically in pure NumPy using subgradient descent.
- The loss function includes both an L1 penalty term (scaled by `alpha * l1_ratio`) and an L2 penalty term (scaled by `alpha * (1 - l1_ratio)`).
- Tested the implementation on a synthetic dataset with two features, one informative and one irrelevant.

**Outcome:**
- The model successfully converged and learned the relationship.
- The fitted weights correctly prioritized the informative feature while penalizing the magnitude of the weights.
- The combination of L1 and L2 allowed for a balance between driving irrelevant feature weights to zero and preventing over-reliance on any single feature.

**Next Steps:**
- Explore tree-based regression models for non-linear relationships.

**Script:** `train_elastic_net_component.py`
