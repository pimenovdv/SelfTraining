# Experiment 0066: Train Temporal Convolutional Network (TCN) Component

## Objective
To implement and verify a Temporal Convolutional Network (TCN) component from scratch using pure NumPy. The goal is to validate the mathematical formulation of causal dilated convolutions, residual blocks, and backpropagation through the network on a sequence modeling task.

## Setup
*   **Script:** `train_tcn_component.py`
*   **Architecture:** 1D Causal Dilated Convolutional Network with 3 residual blocks. Dilation increases by a factor of 2 at each level.
*   **Task:** Sequence delay task (predicting a sequence delayed by 3 steps).
*   **Hyperparameters:** Epochs=1000, Learning Rate=0.01, Sequence Length=15, Hidden Dimension=8

## Execution
The script was executed to train the TCN on the synthetic sequence task.

## Results
*   **Status:** Success
*   **Final Training Loss:** 0.0140
*   **Final Test Loss:** 0.0187

## Observations & Insights
*   The TCN successfully learned to model the temporal dependencies required for the sequence delay task.
*   Causal dilated convolutions effectively allow the model to have a large receptive field (exponentially increasing with depth) while maintaining temporal order without future information leakage.
*   The gradients were successfully propagated backward through time across multiple residual blocks and dilated convolutional layers, confirming the correctness of the manual backpropagation implementation.
