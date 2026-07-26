# Experiment 0035: Train Linear Attention Component

## Objective
To implement and train a small-scale, mathematically rigorous Linear Attention mechanism. Standard self-attention has a time and memory complexity of $O(N^2)$ where $N$ is the sequence length, due to the explicit computation of the $N \times N$ attention matrix. Linear Attention tests the hypothesis that by using a kernel feature map $\phi(x)$ (e.g., ELU + 1) to ensure positivity, we can approximate attention and exploit the associativity of matrix multiplication to compute the output as $\phi(Q) (\phi(K)^T V)$, reducing complexity to $O(N)$.

## Setup
*   **Script:** `train_linear_attention_component.py`
*   **Data:** Synthetic sequence dataset.
*   **Hyperparameters:** `d_k` = 2, `epochs` = 10000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes for Linear Attention. The implementation successfully factored out the $N \times N$ attention matrix computation into $O(N)$ operations.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 10000 epochs.
*   **Predictions:** The final predictions approximate the expected target outputs, verifying gradients flow correctly.

## Observations & Next Steps
*   The implementation correctly demonstrates that linear attention mechanisms can learn sequence mappings without an explicit $N \times N$ attention matrix.
*   Manual derivation of backpropagation using `numpy` solidifies the theoretical understanding of gradient computation for the factored matrix multiplications and the non-linear kernel feature map.
*   Next steps could involve replacing standard attention in our full Transformer blocks with this linear attention to measure actual speedups on longer synthetic sequences.
