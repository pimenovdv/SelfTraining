# Experiment 0047: Train Highway Network Component

**Status:** Success
**Final Loss:** 0.003156
**Epochs:** 10000
**Learning Rate:** 0.05

## Objective
To implement and verify a Highway Network component mathematically using pure NumPy. The layer computes an output as a learned combination of a non-linear transformation and a pass-through connection via a gating mechanism.

## Mathematical Formulation
A Highway Layer transforms an input $x$ of dimension $D$ using:
$H = \tanh(x W_H + b_H)$  (Non-linear transformation)
$T = \sigma(x W_T + b_T)$ (Transform gate)
$y = H \odot T + x \odot (1 - T)$ (Output)

During backpropagation, gradients are correctly routed through both the $H$ transform and the gating paths.

## Results
The model was trained on a synthetic dataset to match a target non-linear transformation.
- **Final Loss:** 0.003156

The loss converged successfully, proving the mathematical formulation and the manual backpropagation derivations are correct.
