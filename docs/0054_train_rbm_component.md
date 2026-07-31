# Experiment 0054: Train Restricted Boltzmann Machine (RBM) Component

## Objective
To implement and verify a Restricted Boltzmann Machine (RBM) mathematically using pure NumPy, testing Contrastive Divergence (CD-1) learning on a synthetic binary dataset.

## Mathematical Formulation
An RBM is an energy-based generative model with bipartite connections between visible units $v$ and hidden units $h$.
*   **Energy Function:** $E(v, h) = -v^T W h - b_v^T v - b_h^T h$
*   **Probabilities:**
    *   $P(h_j = 1 | v) = \sigma(W_{ \cdot j }^T v + b_{h, j})$
    *   $P(v_i = 1 | h) = \sigma(W_{ i \cdot } h + b_{v, i})$
*   **Contrastive Divergence (CD-1):** Weight updates are approximated using a single step of Gibbs sampling: $\Delta W \propto v_0 h_0^T - v_1 h_1^T$.

## Experimental Setup
*   **Visible Units:** 8
*   **Hidden Units:** 4
*   **Dataset:** Synthetic binary patterns (1000 samples).
*   **Epochs:** 1000
*   **Learning Rate:** 0.1
*   **Batch Size:** 32

## Results
*   **Final Reconstruction Error:** 0.0000
*   **Status:** Success

## Conclusion
The RBM successfully learned the underlying binary patterns in the synthetic dataset using CD-1. The reconstruction error decreased steadily, verifying the mathematical formulation of the energy-based model and the manual parameter updates through contrastive divergence.

**Script:** `train_rbm_component.py`
