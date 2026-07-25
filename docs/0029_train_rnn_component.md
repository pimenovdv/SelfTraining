# Experiment 0029: Train RNN Component (Elman Network)

## Objective
To implement and train a simple Recurrent Neural Network (RNN) using pure mathematics to test the hypothesis that a sequential state mechanism can store information over time steps and solve a delayed reasoning task. We test this on a sequential version of the XOR problem.

## Setup
*   **Script:** `train_rnn_component.py`
*   **Data:** Sequential XOR dataset (2 time steps).
*   **Hyperparameters:** `hidden_size` = 8, `epochs` = 50000, `learning_rate` = 1.0

## Execution
The training script was executed to verify the mathematical formulation of the recurrent forward pass and Backpropagation Through Time (BPTT).

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 50000 epochs.
*   **Predictions:** The final predictions correctly compute the XOR of the input across the two time steps, proving that the hidden state successfully retained information from the first step to be combined with the second step.

## Observations & Next Steps
*   The implementation correctly demonstrates sequential memory and processing capabilities.
*   Manual derivation of Backpropagation Through Time (BPTT) using `numpy` solidifies the theoretical understanding of gradient descent in recurrent structures.
*   While self-attention mechanisms (Transformers) are the current paradigm, verifying recurrent memory structures builds the foundation for understanding stateful memory (e.g., Mamba, RNNs) which might be critical for efficient AGI processing over infinite contexts.
