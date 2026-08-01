# Experiment 0039: Sparse Autoencoder (SAE) Component

## Objective
Implement and verify a Sparse Autoencoder (SAE) using pure NumPy. The goal is to mathematically model an autoencoder that learns a sparse, overcomplete representation of the data, which is commonly used in mechanistic interpretability.

## Mathematical Formulation

### Forward Pass
Let $x \in \mathbb{R}^D$ be the input data.
The encoder maps the input to a higher-dimensional hidden space $F$ ($F > D$) with a ReLU activation to encourage non-negativity:
$z = \text{ReLU}(x W_e + b_{enc})$
The decoder attempts to reconstruct the original input:
$\hat{x} = z W_d + b_{dec}$

### Loss Function
The model is trained to minimize the reconstruction error (Mean Squared Error) while encouraging sparsity in the latent representation via an L1 penalty:
$\mathcal{L} = \frac{1}{B \cdot D} \sum_{i,j} (x_{i,j} - \hat{x}_{i,j})^2 + \lambda \frac{1}{B} \sum_{i,k} |z_{i,k}|$

### Backward Pass
Gradients are calculated manually:
$\frac{\partial \mathcal{L}_{MSE}}{\partial \hat{x}} = \frac{2}{B \cdot D} (\hat{x} - x)$
$\frac{\partial \mathcal{L}_{L1}}{\partial z} = \frac{\lambda}{B} \text{sign}(z)$
These are routed back through the decoder and encoder via the chain rule.

## Results
- **Status:** Success
- **Final Total Loss:** 0.2481
- **Final MSE Loss:** 0.0814
- **Final L1 Loss:** 0.1667
- **Average Active Neurons:** 12.7 / 64
- **Epochs:** 10000

## Conclusion
The model successfully learned to reconstruct the input data while maintaining a sparse latent representation, verifying the mathematical soundness of the SAE formulation and its manual backpropagation.
