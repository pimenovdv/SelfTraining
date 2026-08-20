# Decision Tree Component

**Script:** `train_decision_tree_component.py`
**Status:** Success


## Overview
Evaluates a Decision Tree classifier mathematically in pure NumPy.

## Mathematical Foundation
A Decision Tree splits the feature space recursively based on threshold values that maximize Information Gain.
The Information Gain is based on Gini Impurity:
$$ Gini(y) = 1 - \sum_{i=1}^C p_i^2 $$
$$ Information Gain = Gini(Parent) - \sum_{j} \frac{N_j}{N} Gini(Child_j) $$

## Results
The implementation successfully classified a synthetic binary dataset, achieving a high accuracy, verifying the splitting logic and recursive tree building.
