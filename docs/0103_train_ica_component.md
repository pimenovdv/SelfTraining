# Experiment 0103: Train Independent Component Analysis (ICA) Component

## Objective
To implement and verify Independent Component Analysis (ICA) in pure NumPy using the FastICA algorithm. This explores unsupervised representation learning for separating linearly mixed, non-Gaussian source signals (blind source separation), modeling the cocktail party problem.

## Setup
*   **Script:** `train_ica_component.py`
*   **Data:** Synthetic dataset containing two mixed signals (a sine wave and a square wave).
*   **Method:** FastICA with negentropy maximization using a hyperbolic tangent contrast function.

## Execution
The script centers and whitens the mixed data, then applies FastICA fixed-point iteration to recover the unmixing matrix and estimate the original source signals. The quality of separation is measured by the maximum absolute cross-correlation between the true and estimated signals.

## Results
*   **Status:** Success
*   **Convergence:** FastICA converged quickly.
*   **Max Correlations:** 0.9984 and 0.9996

## Observations & Next Steps
*   The implementation successfully unmixed the signals, recovering the original non-Gaussian sources with high accuracy.
*   Whitening (decorrelation and variance normalization) was critical for the stability and speed of the fixed-point iteration.
*   This validates ICA as a powerful tool for discovering hidden factors and supports the goal of exploring biologically plausible and statistical learning methods.
