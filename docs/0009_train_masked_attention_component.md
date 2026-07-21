# Experiment 0009: Train Masked Self-Attention Component

## Objective
To implement and train a small-scale, mathematically rigorous Masked Self-Attention mechanism component of AGI. This tests the hypothesis that a causal mask can effectively restrict attention to previous tokens in a sequence, a requirement for autoregressive models, utilizing pure matrix operations and manual backpropagation.

## Setup
*   **Script:** `train_masked_attention_component.py`
*   **Data:** Synthetic sequence dataset.
*   **Hyperparameters:** `d_k` = 2, `epochs` = 10000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes, including causal masking.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 10000 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs, adhering to causal constraints.

## Observations & Next Steps
*   The implementation correctly demonstrates masked self-attention capabilities.
*   Manual derivation of backpropagation using `numpy` confirms that the masked positions simply receive zero gradients, validating the theoretical understanding of causal modeling.
*   Next steps could involve integrating masked self-attention into a decoder block.
