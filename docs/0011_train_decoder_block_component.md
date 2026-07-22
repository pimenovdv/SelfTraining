# Experiment 0011: Train Decoder Block Component

## Objective
To implement and train a full Decoder Block component of AGI using pure `numpy`. This tests the integration of Masked Self-Attention, Cross-Attention, Feed-Forward Networks, and Layer Normalization, and validates the manual backpropagation through all these combined components and their residual connections.

## Setup
*   **Script:** `train_decoder_block_component.py`
*   **Data:** Synthetic target and source sequence datasets.
*   **Hyperparameters:** `d_model` = 4, `d_k` = 2, `d_ff` = 8, `epochs` = 10, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes for the entire decoder block.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 10 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs, showing the successful integration of causal self-attention, cross-attention, and non-linear transformations.

## Observations & Next Steps
*   The implementation correctly demonstrates full decoder block capabilities.
*   Manual derivation of backpropagation using `numpy` confirms that gradients are properly routed back through all layers, including cross-attention to both source and target representations, and through causal masks without leaking information.
*   Next steps could involve integrating the encoder and decoder blocks into a full Transformer architecture.
