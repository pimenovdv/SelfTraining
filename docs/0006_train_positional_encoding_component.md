# Experiment 0006: Train Positional Encoding Component

## Objective
To implement and mathematically formulate the Positional Encoding (sine/cosine) component. This tests the hypothesis that positional encodings contain linearly separable order information that can be learned by a simple linear layer.

## Setup
*   **Script:** `train_positional_encoding_component.py`
*   **Data:** Synthetic positional encodings for sequence length 10.
*   **Hyperparameters:** `d_model` = 16, `seq_len` = 10, `epochs` = 5000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of positional encodings and the forward/backward passes of a linear layer predicting normalized absolute positions.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 5000 epochs.
*   **Predictions:** The final predictions closely approximate the expected normalized position targets.

## Observations & Next Steps
*   The implementation correctly demonstrates that sine and cosine based positional encodings contain robust positional information that can be linearly extracted.
*   This validates the theoretical underpinning of adding these encodings to input embeddings in sequence models.
*   Next steps could involve integrating positional encodings directly into the input of the previously verified Transformer Block.
