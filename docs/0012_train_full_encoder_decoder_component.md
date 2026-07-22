# Experiment 0012: Train Full Encoder-Decoder Transformer Component

## Objective
To implement and train a full Encoder-Decoder Transformer architecture using pure `numpy`. This tests the end-to-end integration of Encoder blocks (Self-Attention, FFN) and Decoder blocks (Masked Self-Attention, Cross-Attention, FFN), verifying that backpropagation correctly flows through the entire computational graph across both sequences.

## Setup
*   **Script:** `train_full_encoder_decoder_component.py`
*   **Data:** Synthetic source (encoder input) and target (decoder input) sequence datasets.
*   **Hyperparameters:** `d_model` = 4, `d_k` = 2, `d_ff` = 8, `epochs` = 20000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes for the unified encoder-decoder architecture.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 20000 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs, confirming that the full pipeline correctly translates representations from the source domain to the target domain.

## Observations & Next Steps
*   The implementation correctly demonstrates a fully functional, minimal mathematical model of the original Transformer architecture.
*   Gradients are successfully routed from the decoder output, back through the cross-attention, into the encoder's contextualized representations, and all the way back to the encoder's self-attention layers.
*   Next steps could involve implementing more advanced structural components (like RoPE or SwiGLU) or scaling laws.
