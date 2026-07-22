# Experiment 0015: Train RoPE Component

## Objective
To implement and mathematically formulate Rotary Positional Embeddings (RoPE). This tests the hypothesis that RoPE can effectively inject relative positional information into attention scores by rotating query and key representations, verifiable through manual backpropagation.

## Setup
*   **Script:** `train_rope_component.py`
*   **Data:** Synthetic random input sequence. The target is a relative attention pattern (e.g., high attention to adjacent tokens).
*   **Hyperparameters:** `d_model` = 16, `seq_len` = 10, `epochs` = 5000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of RoPE forward and backward passes.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 5000 epochs.
*   **Predictions:** The final attention scores closely approximate the target relative distance pattern, showing that queries and keys successfully learned to utilize the injected rotary positional embeddings to form relative attention.

## Observations & Next Steps
*   The implementation validates the theoretical underpinning of RoPE, showing that it preserves relative distances through vector rotation in the complex plane (or 2D real sub-planes) and that backpropagation smoothly flows through this trigonometric transformation.
*   Next steps could involve integrating RoPE directly into the multi-head attention component, replacing standard absolute positional encodings.
