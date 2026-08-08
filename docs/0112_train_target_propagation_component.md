# Experiment 0112: Train Difference Target Propagation (DTP) Component

## Objective
To implement and verify Difference Target Propagation (DTP) from scratch using pure NumPy. DTP is a biologically plausible alternative to backpropagation that trains neural networks without requiring symmetric weight matrices or continuous gradients, by using autoencoders to propagate target activations rather than gradients.

## Mathematical Basis
In backpropagation, errors are propagated using the transpose of the forward weights ($W^T$). In DTP, each layer learns an inverse function $g$ (a backward model) parameterized by separate weights, trained as an autoencoder to invert the forward mapping $f$:
$\min_{W_b} || x - g(f(x) + \epsilon) ||^2$

Targets are propagated backwards. Given a target $t_{i}$ for layer $i$, the target for layer $i-1$ is computed using the inverse function, corrected for the inversion error:
$t_{i-1} = h_{i-1} - g_i(h_i) + g_i(t_i)$
The forward weights are then updated to minimize $|| f_i(h_{i-1}) - t_i ||^2$.

## Implementation Details
- Implemented `TargetPropagationLayer` containing independent forward and backward weights.
- Trained a 3-layer network on a non-linear continuous mapping task.
- Replaced standard backpropagation through hidden layers with target propagation and local forward updates.

## Results
- Successfully trained the network to fit the non-linear function without backpropagating gradients through hidden layers.
- Demonstrated that local inverse models can effectively assign credit in deep architectures.
- **Script:** `train_target_propagation_component.py`

## Status
Success. The component correctly learned targets using independent backward models, providing a working biologically-inspired credit assignment mechanism.
