# Experiment 0004: Train Layer Normalization Component

## Objective
To implement and train a mathematically rigorous Layer Normalization component. This serves to test the hypothesis that layer normalization parameters (gamma and beta) can be learned using basic matrix operations and manual backpropagation to match target distributions.

## Setup
*   **Script:** `train_layernorm_component.py`
*   **Data:** Synthetic dataset (3 samples, 4 features).
*   **Hyperparameters:** `epochs` = 10000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 10000 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs, by learning the scale (gamma) and shift (beta) parameters.

## Observations & Next Steps
*   The implementation correctly demonstrates the ability to normalize features and learn affine transformations via gamma and beta.
*   Manual derivation of backpropagation using `numpy` solidifies the theoretical understanding of gradient descent for normalization layers.
*   Next steps could involve integrating this component with the FFN and Attention components to construct a full Transformer block.
