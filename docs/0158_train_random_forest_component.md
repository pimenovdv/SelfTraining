# Random Forest Component

**Script:** `train_random_forest_component.py`

## Overview
Evaluates a Random Forest classifier mathematically in pure NumPy.

## Mathematical Foundation
A Random Forest is an ensemble of Decision Trees, trained on bootstrap samples of the dataset. It introduces additional randomness by selecting a random subset of features for each split. The final prediction is made by aggregating (majority voting) the predictions of all individual trees. This reduces overfitting and improves generalization compared to a single Decision Tree.

## Results
The implementation successfully classified a synthetic binary dataset using an ensemble of trees, achieving a high accuracy and verifying the bootstrapping and majority voting logic.
