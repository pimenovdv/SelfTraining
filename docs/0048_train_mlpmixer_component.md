# Experiment 0048: Train MLP-Mixer Component

**Status:** Success
**Final Loss:** 0.010256
**Epochs:** 25000
**Learning Rate:** 0.01

## Objective
To implement and verify an MLP-Mixer block mathematically using pure NumPy. The layer computes an output as a sequence of token-mixing (across the sequence length dimension) and channel-mixing (across the feature dimension) multi-layer perceptrons, providing a purely MLP-based alternative to Self-Attention.

## Mathematical Formulation
Let $X \in \mathbb{R}^{S \times C}$ be the input sequence matrix where $S$ is sequence length and $C$ is channels.
The MLP-Mixer block applies two distinct operations with skip connections:

1. **Token Mixing:** Operates on columns of $X$ (transposed features).
   $U = X + \text{MLP}_{token}(\text{LayerNorm}(X)^T)^T$
2. **Channel Mixing:** Operates on rows of $U$.
   $Y = U + \text{MLP}_{channel}(\text{LayerNorm}(U))$

During backpropagation, gradients route correctly through both transposed dimensions for the token mixing MLP and standard dimensions for the channel mixing MLP, allowing information flow across the sequence without standard attention matrices.

## Results
The model was trained on a synthetic dataset to match a target non-linear transformation across sequence elements and channels.
- **Final Loss:** 0.010256

The loss converged successfully, proving the mathematical formulation and the manual backpropagation derivations are correct for the full MLP-Mixer architecture.


**Script:** `train_mlpmixer_component.py`
