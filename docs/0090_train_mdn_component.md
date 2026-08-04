# Experiment: Mixture Density Network (MDN)

**Script:** `train_mdn_component.py`
**Date:** 2024-08-04
**Status:** Success

## Description
Evaluated a Mixture Density Network (MDN) component using pure NumPy. The script implements an MDN to predict a multi-modal conditional probability distribution $p(y|x)$ using a Gaussian Mixture Model output layer.

## Methodology
- **Architecture:** One hidden layer MLP mapping inputs to the parameters (mixing coefficients, means, and variances) of 5 Gaussians.
- **Task:** Learning an inverse kinematics toy problem where a single input $x$ can map to multiple valid outputs $y$.
- **Optimization:** Minimized the Negative Log-Likelihood (NLL) of the Gaussian Mixture using gradient descent.

## Results
- The network successfully minimized the NLL.
- Initial Loss: 1.0935
- Final Loss: -0.1424
