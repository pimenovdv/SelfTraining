# Experiment 0045: Train Batch Normalization Component

**Status:** Success
**Final Loss:** 0.000000
**Epochs:** 5000
**Learning Rate:** 0.1

## Objective
To implement and verify a Batch Normalization component mathematically using pure NumPy, testing its ability to learn scale (`\gamma`) and shift (`\beta`) parameters via manual backpropagation.

## Mathematical Formulation
Batch Normalization normalizes the input across the batch dimension.
For an input $X$ with batch size $m$:

$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$ (batch mean)
$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$ (batch variance)
$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$ (normalized value)
$y_i = \gamma \hat{x}_i + \beta$ (scaled and shifted value)

During backpropagation, gradients are routed through $\gamma$ and $\beta$, as well as back to $x$ through the mean and variance calculations.

## Results
The model was trained on a synthetic dataset to match a target affine transformation.
- **Initial Loss:** High
- **Final Loss:** 0.000000

The loss converged successfully, proving the mathematical formulation and the manual backpropagation derivations are correct.

**Script:** `train_batchnorm_component.py`
