# Experiment: Weight Normalization Component

**Script:** `train_weight_normalization_component.py`

## Objective
To implement and mathematically formalize Weight Normalization, a reparameterization of the weight vectors in a neural network that decouples the length of those weight vectors from their direction.

## Methodology
1.  **Component:** WeightNormLinear
2.  **Algorithm:** Weight matrix $w$ is reparameterized as $w = (g / ||v||) * v$, where $v$ is a parameter vector and $g$ is a scalar parameter.
3.  **Forward Pass:** Calculate norm of $v$, scale by $g/||v||$, and perform linear transformation.
4.  **Backward Pass:** Exact gradient computation for both $v$ and $g$ using the chain rule.
5.  **Task:** Binary classification using a two-layer weight-normalized network.

## Results
- **Success:** Yes

## Conclusion
Weight Normalization accelerates convergence similar to batch normalization but does not introduce dependencies between examples in a minibatch, making it suitable for recurrent models and noise-sensitive applications like reinforcement learning.
