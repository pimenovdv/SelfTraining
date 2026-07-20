# Experiment 0002: Train 2-Layer FFN Component

## Objective
To implement and train a small-scale, mathematically rigorous Feed-Forward Network (FFN) component of AGI. This serves to test the hypothesis that a simple 2-layer FFN with non-linear activation can learn non-linear reasoning boundaries, such as the XOR problem, using basic matrix operations and manual backpropagation.

## Setup
*   **Script:** `train_ffn_component.py`
*   **Data:** Synthetic XOR dataset.
*   **Hyperparameters:** `hidden_size` = 4, `epochs` = 50000, `learning_rate` = 1.0

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 50000 epochs.
*   **Predictions:** The final predictions closely approximate the expected XOR outputs (0 for identical inputs, 1 for different inputs).

## Observations & Next Steps
*   The implementation correctly demonstrates non-linear transformation capabilities.
*   Manual derivation of backpropagation using `numpy` solidifies the theoretical understanding of gradient descent.
*   Next steps could involve testing deeper architectures, alternative activation functions (e.g., ReLU, GELU), or applying the FFN component to more complex synthetic reasoning tasks before attempting integration into a Transformer architecture.
