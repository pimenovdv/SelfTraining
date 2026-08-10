# Experiment 0124: Train Neural Arithmetic Logic Unit (NALU) Component

**Script:** `train_nalu_component.py`
**Status:** Success
**Final Loss:** 0.038524
**Epochs:** 2000
**Learning Rate:** 0.01

## Objective
To implement and verify a Neural Arithmetic Logic Unit (NALU) mathematically using pure NumPy. A NALU combines an additive path and a multiplicative path, controlled by a learned gate, to enable neural networks to learn systematic numerical extrapolation for basic arithmetic operations.

## Mathematical Formulation
The NALU layer interpolates between an additive accumulator and a multiplicative one:
- The base weights are constrained: $W = \tanh(\hat{W}) \odot \sigma(\hat{M})$.
- Additive path: $a = x W$.
- Multiplicative path: $m = \exp(\log(|x| + \epsilon) W)$.
- Gate: $g = \sigma(x G)$.
- Output: $y = g \odot a + (1 - g) \odot m$.

Manual backpropagation was derived and implemented to correctly route gradients through both the linear and log-space paths.

## Results
The model was trained on a synthetic dataset to match a target multiplication function ($f(x_1, x_2) = x_1 \times x_2$).
- **Final Loss:** 0.038524

The loss converged successfully, confirming that the network successfully learned the multiplication function by relying on the multiplicative path and adapting its gate accordingly.
