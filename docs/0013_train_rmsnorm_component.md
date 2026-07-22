# Experiment 0013: Train RMSNorm Component

## Objective
To implement and train a mathematically rigorous Root Mean Square Normalization (RMSNorm) component. This serves to test the hypothesis that removing mean-centering (compared to LayerNorm) still allows the model to learn a stable scale parameter (gamma) via manual backpropagation to match target distributions, while being computationally simpler.

## Setup
*   **Script:** `train_rmsnorm_component.py`
*   **Data:** Synthetic dataset (3 samples, 4 features).
*   **Hyperparameters:** `epochs` = 10000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 10000 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs by learning the scale (gamma) parameter alone without a shift (beta) or mean-centering.

## Observations & Next Steps
*   The implementation correctly demonstrates the ability to normalize features using RMS and learn scaling transformations via gamma.
*   Manual derivation of backpropagation using `numpy` confirms that RMSNorm is computationally simpler and its gradients are properly routed.
*   Next steps could involve comparing its convergence rate with standard LayerNorm in a full Transformer block.
