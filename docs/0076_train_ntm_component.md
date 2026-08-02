# Experiment 0076: Train Neural Turing Machine (NTM) Component

## Objective
Implement and mathematically model a Neural Turing Machine (NTM) component, testing the hypothesis that differentiable external memory can be addressed via content similarity and selectively read from and written to using backpropagation.

## Setup
*   **Script:** `train_ntm_component.py`
*   **Data:** Synthetic key-value retrieval task.
*   **Hyperparameters:** `epochs` = 1000, `learning_rate` = 0.5, `N` (memory slots) = 4, `d` (memory dimension) = 8

## Execution
The script was executed to verify the mathematical formulation of content-based addressing (cosine similarity + softmax) and the manual backpropagation of gradients through the memory read mechanism to optimize a query vector.

## Results
*   **Status:** Success
*   **Final Loss:** 0.0010
