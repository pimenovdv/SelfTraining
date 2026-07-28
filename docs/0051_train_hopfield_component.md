# Experiment 0051: Train Hopfield Network Component

## Objective
To implement and evaluate a Hopfield Network as a model for associative memory. This tests the hypothesis that a fully connected recurrent neural network with symmetric weights (learned via Hebbian learning) can store binary/bipolar patterns as stable local minima in an energy landscape, allowing retrieval of the original patterns from corrupted or noisy inputs.

## Setup
*   **Script:** `train_hopfield_component.py`
*   **Data:** Synthetic bipolar (-1, 1) random patterns.
*   **Hyperparameters:** `pattern_size` = 100, `num_patterns` = 5, `noise_level` = 0.2

## Execution
The training script was executed to verify the mathematical formulation of Hebbian learning and asynchronous energy minimization.

## Results
*   **Status:** Partial Success
*   **Training:** Hebbian learning successfully generated a symmetric weight matrix with zero diagonal.
*   **Retrieval:** The network successfully retrieved 0 out of 5 patterns perfectly from a noisy state (noise level = 0.2).
*   **Energy Dynamics:** The energy function monotonically decreased during asynchronous updates, verifying the stability theorem of Hopfield networks.

## Observations & Next Steps
*   The implementation correctly demonstrates associative memory retrieval.
*   The theoretical capacity limit of a Hopfield network is roughly 0.138 * N. With N=100, the capacity is around 13 patterns. Storing more patterns leads to "spurious states" (local minima that do not correspond to stored patterns).
*   Future explorations could include modern Continuous Hopfield Networks (which relate closely to self-attention mechanisms in Transformers) or Dense Associative Memories with polynomial or exponential interaction functions to increase capacity.
