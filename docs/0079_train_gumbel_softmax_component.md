# Experiment 0079: Train Gumbel-Softmax Component

## Objective
Implement and mathematically model a Gumbel-Softmax estimator, testing the hypothesis that the reparameterization trick with Gumbel noise allows differentiable discrete sampling from a categorical distribution, enabling training via manual backpropagation.

## Setup
*   **Script:** `train_gumbel_softmax_component.py`
*   **Data:** Synthetic target categorical distribution (one-hot).
*   **Hyperparameters:** `epochs` = 5000, `learning_rate` = 0.05, `num_classes` = 4

## Execution
The script was executed to verify the mathematical formulation of Gumbel-Softmax sampling, temperature annealing, and the manual backpropagation of gradients to update underlying logits.

## Results
*   **Status:** Success
*   **Final Loss:** -0.0000
*   **Final Probabilities:** [3.000e-04 1.000e-04 9.994e-01 2.000e-04]
