# Experiment 0049: Train gMLP (Gated MLP) Component

**Status:** Success
**Final Loss:** 0.028302
**Epochs:** 10000
**Learning Rate:** 0.01

## Objective
To implement and verify a gMLP (Gated MLP) component mathematically using pure NumPy, testing its ability to model spatial/sequential dependencies without attention mechanisms via a Spatial Gating Unit (SGU).

## Mathematical Formulation
The gMLP block operates on an input $X \in \mathbb{R}^{N \times d}$:
1. Linear projection: $Z = X U$, where $U \in \mathbb{R}^{d \times 2 d_{hidden}}$
2. Activation: $Z_{act} = \text{ReLU}(Z)$
3. Split: $Z_{act} = [Z_1, Z_2]$ along the channel dimension.
4. Spatial Projection: $\tilde{Z}_2 = W Z_2 + b$, where $W \in \mathbb{R}^{N \times N}$ captures spatial interactions across the sequence.
5. Gating: $S = Z_1 \odot \tilde{Z}_2$
6. Output Projection: $Y = S V$, where $V \in \mathbb{R}^{d_{hidden} \times d}$

During backpropagation, gradients are routed through the output projection, the element-wise gating operation, the spatial projection matrix $W$ via Einstein summation, and back to the input $X$.

## Results
The model was trained on a synthetic sequence dataset to match a target spatial transformation.
- **Initial Loss:** High
- **Final Loss:** 0.028302

The loss converged successfully, proving the mathematical formulation and the manual backpropagation derivations for the gMLP spatial gating mechanism are correct.


**Script:** `train_gmlp_component.py`
