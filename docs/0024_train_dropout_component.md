# Experiment 0024: Train Dropout Component

## Objective
To implement and train a small-scale Feed-Forward Network (FFN) utilizing Inverted Dropout. This component tests the hypothesis that randomly dropping neuron activations during training reduces overfitting by preventing complex co-adaptations. It verifies the mathematical soundness of applying dropout masks and scaling forward/backward passes appropriately.

## Setup
*   **Script:** `train_dropout_component.py`
*   **Data:** Synthetic XOR dataset.
*   **Hyperparameters:** `hidden_size` = 16, `epochs` = 100000, `learning_rate` = 1.0, `drop_rate` = 0.2

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes for inverted dropout.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 100000 epochs, despite the noisy gradients introduced by dropout.
*   **Predictions:** The final predictions at inference time (without dropout) closely approximate the expected XOR outputs (0 for identical inputs, 1 for different inputs).

## Observations & Next Steps
*   The Inverted Dropout implementation correctly demonstrates its regularizing capabilities without altering inference-time computations.
*   Manual derivation of backpropagation through the dropout mask and inverted scaling solidifies the mathematical framework.
*   Next steps could involve integrating Dropout into complex architectures like the full Transformer Block to stabilize training on larger datasets.
