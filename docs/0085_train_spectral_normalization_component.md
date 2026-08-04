# Experiment: Spectral Normalization Component

**Script:** `train_spectral_normalization_component.py`

## Objective
To implement and mathematically formalize Spectral Normalization for deep neural networks using power iteration, enabling Lipschitz continuity for more stable training.

## Methodology
1.  **Component:** SpectralNormLinear
2.  **Algorithm:** Power Iteration to find the largest singular value $\sigma$ of weight matrix $W$.
3.  **Forward Pass:** Weight matrix is scaled by $1/\sigma$.
4.  **Backward Pass:** Exact gradient computation incorporating the derivative of $\sigma$ with respect to $W$.
5.  **Task:** Binary classification using a two-layer spectral normalized network.

## Results
- **Success:** Yes

## Conclusion
Spectral Normalization provides an effective way to enforce Lipschitz constraints, mathematically grounding regularization techniques for advanced generative and discriminative models.
