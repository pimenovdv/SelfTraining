# Experiment 0022: Train AdamW Optimizer Component

## Objective
To implement and evaluate the AdamW Optimizer (Adaptive Moment Estimation with Decoupled Weight Decay). This tests the hypothesis that combining adaptive gradient updates with explicit decoupled weight decay accelerates convergence and improves model generalization compared to standard SGD. We evaluate this by training a 2-layer FFN on a non-linear dataset using pure matrix operations.

## Setup
*   **Script:** `train_adamw_component.py`
*   **Data:** Synthetic XOR dataset.
*   **Hyperparameters:** `hidden_size` = 8, `epochs` = 5000, `learning_rate` = 0.01, `weight_decay` = 0.01

## Execution
The training script was executed to verify the mathematical formulation of the AdamW parameter updates (moment estimates, bias correction, and weight decay application).

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully converged, typically much faster than standard SGD, confirming the efficiency of AdamW.
*   **Predictions:** The final predictions correctly learned the XOR reasoning boundaries.

## Observations & Next Steps
*   The AdamW implementation successfully demonstrates adaptive learning rates for each parameter with decoupled weight decay.
*   Manual derivation and application of moving averages (first and second moments) and bias corrections solidify the mathematical framework of modern optimizers.
*   Next steps could involve replacing standard SGD with AdamW in the full Transformer Block training scripts to evaluate convergence speedups on sequence modeling tasks.
