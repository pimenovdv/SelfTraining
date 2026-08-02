# Experiment 0077: Train Skip-Gram Component

## Objective
Implement and mathematically model a Skip-Gram component with Negative Sampling, testing the hypothesis that word representations can be learned by maximizing the similarity between target words and their contexts while minimizing similarity with negative samples via manual backpropagation.

## Setup
*   **Script:** `train_skipgram_component.py`
*   **Data:** Synthetic word context pairs.
*   **Hyperparameters:** `epochs` = 5000, `learning_rate` = 0.1, `V` (vocab) = 5, `d` (embed_dim) = 4

## Execution
The script was executed to verify the mathematical formulation of Skip-Gram Negative Sampling and the manual backpropagation of gradients to update target and context embeddings.

## Results
*   **Status:** Success
*   **Final Loss:** 0.0004
