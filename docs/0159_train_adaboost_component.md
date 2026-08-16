# AdaBoost Component

**Script:** `train_adaboost_component.py`

## Overview
Evaluates an AdaBoost classifier mathematically in pure NumPy.

## Mathematical Foundation
AdaBoost (Adaptive Boosting) is an ensemble learning method that combines multiple weak classifiers (e.g., decision stumps) to create a strong classifier. It sequentially trains the weak classifiers, iteratively updating the sample weights so that subsequent classifiers focus more on the samples that were misclassified by the previous ones.
The final prediction is a weighted sum of the weak classifiers' predictions.
$$ H(x) = \text{sign}\left( \sum_{t=1}^T \alpha_t h_t(x) \right) $$
where $\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$.

## Results
The implementation successfully classified a synthetic binary dataset using an ensemble of decision stumps, achieving high accuracy and verifying the iterative weighting logic.
