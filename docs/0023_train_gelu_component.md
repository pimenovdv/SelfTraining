# Experiment 0023: Train GELU Component

## Objective
To implement and train a small-scale Feed-Forward Network (FFN) utilizing the Gaussian Error Linear Unit (GELU) activation function. This component tests the hypothesis that advanced activation functions provide richer representational capacity than standard ReLUs or Sigmoids. We test its ability to learn non-linear reasoning boundaries (e.g., XOR) using pure matrix operations and manual backpropagation.

## Setup
*   **Script:** `train_gelu_component.py`
*   **Data:** Synthetic XOR dataset.
*   **Hyperparameters:** `hidden_size` = 8, `epochs` = 50000, `learning_rate` = 1.0

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes for the GELU activation.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 50000 epochs.
*   **Predictions:** The final predictions closely approximate the expected XOR outputs (0 for identical inputs, 1 for different inputs).

## Observations & Next Steps
*   The GELU implementation correctly demonstrates non-linear transformation capabilities.
*   Manual derivation of backpropagation, particularly for the GELU approximation and its derivative, solidifies the mathematical framework.
*   Next steps could involve integrating GELU into the full Transformer Block or exploring other advanced activation functions.
