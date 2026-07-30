# Experiment 0058: Train Spiking Neural Network (SNN) Component

## Objective
To implement and train a Spiking Neural Network (SNN) with Leaky Integrate-and-Fire (LIF) neurons in pure NumPy. This serves to verify the mathematical formulation of spiking dynamics (membrane potential integration, firing threshold, reset) and manual backpropagation using Surrogate Gradients to overcome the non-differentiable spiking step function.

## Setup
*   **Script:** `train_snn_component.py`
*   **Architecture:** Input (2) -> LIF Layer (32) -> Output Rate Decoding (1)
*   **Data:** XOR problem presented as a constant current over `10` time steps.
*   **Hyperparameters:** `epochs` = 2000, `lr` = 5.0, `hidden_dim` = 32, `T` = 10
*   **Surrogate Function:** Fast Sigmoid `1 / (1 + alpha * |x|)^2` with `alpha=10.0`

## Execution
The training script was executed to verify the forward and backward passes of the LIF network using BPTT and surrogate gradients.

## Results
*   **Status:** Success
*   **Final Loss:** 0.0002
*   **Performance:** The SNN successfully minimized the binary cross-entropy loss, learning the non-linear XOR boundary using event-based spikes and mean firing rate decoding.

## Observations & Next Steps
*   The implementation correctly demonstrates the integration of surrogate gradients into Backpropagation Through Time (BPTT), validating its mathematical soundness.
*   Next steps could involve testing on more complex sequential datasets, analyzing energy efficiency via spike sparsity, or implementing different reset mechanisms (soft reset).
