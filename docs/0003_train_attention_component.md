# Experiment 0003: Train Self-Attention Component

## Objective
To implement and train a small-scale, mathematically rigorous Self-Attention mechanism component of AGI. This serves to test the hypothesis that a single-head self-attention layer can learn relationships between elements in a sequence using basic matrix operations and manual backpropagation.

## Setup
*   **Script:** `train_attention_component.py`
*   **Data:** Synthetic sequence dataset.
*   **Hyperparameters:** `d_k` = 2, `epochs` = 10000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 10000 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs.

## Observations & Next Steps
*   The implementation correctly demonstrates self-attention mechanism capabilities.
*   Manual derivation of backpropagation using `numpy` solidifies the theoretical understanding of gradient descent for attention mechanisms.
*   Next steps could involve testing multi-head attention, scaling the dimensions, or integrating this component with the previously built FFN component to form a complete Transformer block.
