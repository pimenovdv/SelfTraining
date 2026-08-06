# Experiment 0102: Train Hebbian Learning Component (Oja's Rule)

## Objective
To implement and verify biologically plausible Hebbian learning using Oja's rule in pure NumPy. This explores unsupervised, gradient-free learning rules where weight updates depend only on local pre-synaptic and post-synaptic activities, demonstrating its mathematical equivalence to finding the principal component of the input data.

## Setup
*   **Script:** `train_hebbian_component.py`
*   **Data:** Synthetic 2D dataset with a strong principal axis.
*   **Hyperparameters:** `epochs` = 1000, `learning_rate` = 0.01

## Execution
The training script was executed to update a single neuron's weights using Oja's rule (a stable variant of Hebbian learning). The learned weights were then compared against the theoretical first principal component calculated via Singular Value Decomposition (SVD).

## Results
*   **Status:** Success
*   **Final Reconstruction Error:** 0.0035
*   **Learned Vector:** [0.4445, 0.8958]
*   **Theoretical PC1:** [0.4405, 0.8977]
*   **Cosine Similarity:** 0.999990

## Observations & Next Steps
*   The implementation successfully converged to the first principal component without using backpropagation or gradient descent.
*   Oja's rule effectively balances standard Hebbian growth ($y \cdot x$) with a weight decay term ($y^2 \cdot W$), ensuring stability.
*   Next steps could involve implementing Generalized Hebbian Algorithm (Sanger's rule) for extracting multiple principal components, or applying Hebbian updates in competitive networks.
