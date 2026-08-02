# Experiment 0078: Train Continuous Bag of Words (CBOW) Component

## Objective
Implement and mathematically model a Continuous Bag of Words (CBOW) component, testing the hypothesis that word representations can be learned by predicting a target word from the average of its context word embeddings, utilizing manual backpropagation.

## Setup
*   **Script:** `train_cbow_component.py`
*   **Data:** Synthetic context-target word pairs.
*   **Hyperparameters:** `epochs` = 2000, `learning_rate` = 0.1, `V` (vocab) = 5, `d` (embed_dim) = 4

## Execution
The script was executed to verify the mathematical formulation of CBOW and the manual backpropagation of gradients to update context and target embeddings.

## Results
*   **Status:** Success
*   **Final Loss:** 0.0024
