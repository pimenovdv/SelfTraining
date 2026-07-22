# Experiment 0010: Train Cross-Attention Component

## Objective
To implement and train a small-scale, mathematically rigorous Cross-Attention mechanism component of AGI. This tests the hypothesis that a cross-attention layer can learn relationships between a target sequence (queries) and a source sequence (keys, values) using basic matrix operations and manual backpropagation.

## Setup
*   **Script:** `train_cross_attention_component.py`
*   **Data:** Synthetic target and source sequence datasets.
*   **Hyperparameters:** `d_k` = 2, `epochs` = 10000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes for cross-attention.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 10000 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs, effectively attending to the source sequence based on target queries.

## Observations & Next Steps
*   The implementation correctly demonstrates cross-attention mechanism capabilities.
*   Manual derivation of backpropagation using `numpy` confirms that gradients are properly routed back to both the target sequence representation (via Q) and source sequence representation (via K and V).
*   Next steps could involve integrating masked self-attention and cross-attention into a full decoder block.
