# Experiment 0020: Train ALiBi Component

## Objective
To implement and train a small-scale, mathematically rigorous Attention with Linear Biases (ALiBi) mechanism component of AGI. This serves to test the hypothesis that positional information can be effectively injected directly into attention scores without learning embeddings, utilizing pure matrix operations and manual backpropagation.

## Setup
*   **Script:** `train_alibi_component.py`
*   **Data:** Synthetic sequence dataset.
*   **Hyperparameters:** `d_model` = 4, `d_k` = 2, `heads` = 2, `epochs` = 10000, `learning_rate` = 0.01

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes, including the ALiBi bias and causal masking.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 10000 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs, adhering to causal constraints and leveraging relative distance biases.

## Observations & Next Steps
*   The implementation correctly demonstrates ALiBi mechanism capabilities without the need for positional embeddings (e.g. RoPE or sinusoidal).
*   Manual derivation of backpropagation using `numpy` confirms that the ALiBi bias, lacking learnable parameters, acts as a constant during backpropagation to scores and safely routes gradients, validating its theoretical formulation.
*   Next steps could involve integrating ALiBi into a full Transformer block or comparing its extrapolation capabilities with RoPE.
