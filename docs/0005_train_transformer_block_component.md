# Experiment 0005: Train Transformer Block Component

## Objective
To implement and train a full mathematically rigorous single-layer Transformer Block. This tests the hypothesis that we can successfully combine the Self-Attention mechanism, Layer Normalization, Feed-Forward Network, and residual connections into a unified block and optimize it via manual backpropagation.

## Setup
*   **Script:** `train_transformer_block_component.py`
*   **Data:** Synthetic sequence dataset.
*   **Hyperparameters:** `d_model` = 4, `d_k` = 2, `d_ff` = 8, `epochs` = 20000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes across all components together.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error to near zero (0.0000) over 20000 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs.

## Observations & Next Steps
*   The implementation correctly demonstrates that a Transformer Block comprised of mathematically verified components can learn representations and update via a complex gradient chain.
*   Manual derivation of backpropagation using `numpy` through LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual solidifies the theoretical understanding of full block gradient flow.
*   Next steps could involve stacking multiple blocks, introducing positional encodings, and testing on actual text data with the BPE tokenizer.