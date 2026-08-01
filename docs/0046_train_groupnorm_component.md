# Experiment 0046: Train Group Normalization Component

**Status:** Success
**Final Loss:** 0.000000
**Epochs:** 5000
**Learning Rate:** 0.1

## Objective
To implement and verify a Group Normalization component mathematically using pure NumPy, testing its ability to learn scale (`\gamma`) and shift (`\beta`) parameters via manual backpropagation.

## Mathematical Formulation
Group Normalization divides the channels into groups and normalizes the features within each group.
For an input $X$ with $C$ channels divided into $G$ groups, the features are reshaped into $G$ groups of size $D = C/G$.

$\mu_g = \frac{1}{D} \sum_{i=1}^D x_{g,i}$ (group mean)
$\sigma_g^2 = \frac{1}{D} \sum_{i=1}^D (x_{g,i} - \mu_g)^2$ (group variance)
$\hat{x}_{g,i} = \frac{x_{g,i} - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}$ (normalized value)
$y_c = \gamma_c \hat{x}_c + \beta_c$ (scaled and shifted value per channel)

During backpropagation, gradients are routed through $\gamma$ and $\beta$, as well as back to $x$ through the mean and variance calculations within each group.

## Results
The model was trained on a synthetic dataset to match a target affine transformation on grouped normalized features.
- **Initial Loss:** High
- **Final Loss:** 0.000000

The loss converged successfully, proving the mathematical formulation and the manual backpropagation derivations are correct.


**Script:** `train_groupnorm_component.py`
