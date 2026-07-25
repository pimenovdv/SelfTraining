# Experiment 0031: Train LSTM Component (Long Short-Term Memory)

## Objective
To implement and train a Long Short-Term Memory (LSTM) cell using pure mathematics to test the hypothesis that advanced cell state mechanics (forget, input, output gates) effectively mitigate the vanishing gradient problem and allow for robust sequential memory retention over time steps. We evaluate this on a sequential version of the XOR problem.

## Setup
*   **Script:** `train_lstm_component.py`
*   **Data:** Sequential XOR dataset (2 time steps).
*   **Hyperparameters:** `hidden_size` = 8, `epochs` = 50000, `learning_rate` = 1.0

## Execution
The training script was executed to verify the mathematical formulation of the LSTM forward pass and Backpropagation Through Time (BPTT).

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 50000 epochs.
*   **Predictions:** The final predictions correctly compute the XOR of the input across the two time steps, proving that the cell state and gating mechanisms successfully coordinate to store relevant information across time.

## Observations & Next Steps
*   The implementation confirms that complex cell states and multiple gating mechanisms can be successfully modeled and trained using basic matrix algebra and manual derivation of gradients.
*   Compared to the simple Elman RNN and GRU, the LSTM explicitly models a separate cell state, adding more robust flow control through forget and input gates.
*   Future work might look into integrating continuous-time dynamics or other advanced state-space components.
