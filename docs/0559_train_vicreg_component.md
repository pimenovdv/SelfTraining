**Script:** `train_vicreg_component.py`

# VICReg Component Testing

## Objective
Implement and evaluate the VICReg (Variance-Invariance-Covariance Regularization) Loss component mathematically in pure NumPy.

## Method
Implemented the VICReg loss which prevents representation collapse in self-supervised learning without requiring architectural tricks like stop-gradients or negative examples. The loss consists of three terms:
1. **Invariance:** Mean squared error between augmented views.
2. **Variance:** Hinge loss to maintain standard deviation of representations above a threshold, preventing collapse.
3. **Covariance:** Penalizes non-zero off-diagonal elements of the covariance matrix to decorrelate features and maximize information content.

## Outcome
The implementation successfully verified the expected behavior. Collapsed representations (all zeros) resulted in a high loss due to the variance penalty, while decorrelated representations with high variance yielded a very low loss, demonstrating VICReg's ability to avoid representation collapse while aligning similar inputs.
