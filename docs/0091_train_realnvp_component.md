# Experiment: RealNVP Normalizing Flow

**Script:** `train_realnvp_component.py`
**Date:** 2024-08-04
**Status:** Success

## Description
Evaluated a RealNVP (Real Non-Volume Preserving) component using pure NumPy. The script implements an invertible normalizing flow to map complex data distributions to a simple base distribution (Gaussian).

## Methodology
- **Architecture:** Stack of affine coupling layers with masked networks to maintain an easily computable Jacobian determinant and invertibility.
- **Task:** Learning the mapping for a simple correlated 2D Gaussian dataset to an uncorrelated isotropic Gaussian.
- **Optimization:** Maximizing the Log-Likelihood of the data using backpropagation through the coupling layers.

## Results
- The network successfully minimized the Negative Log-Likelihood (NLL).
- Initial Loss: 2.9421
- Final Loss: 1.2300
