# Experiment 0036: Train Retention Component

## Objective
To implement and train a Retention Mechanism (from RetNet) in pure NumPy. This serves to test the hypothesis that we can create a sequence model that supports both parallel training and $O(1)$ recurrent inference, bridging the gap between Transformers and RNNs.

## Setup
*   **Script:** `train_retention_component.py`
*   **Data:** Synthetic sequence dataset.
*   **Hyperparameters:** `d_model` = 16, `epochs` = 1000, `learning_rate` = 0.01, `gamma` = 0.9

## Execution
The training script was executed to verify the mathematical formulation of the parallel forward and manual backward passes. Additionally, the $O(1)$ recurrent formulation was verified against the parallel formulation.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 1000 epochs.
*   **Recurrent Verification:** The max difference between the parallel and recurrent outputs was 0.00000000, confirming mathematical equivalence.

## Observations & Next Steps
*   The Retention mechanism correctly learns sequence transformations.
*   The explicitly derived backward pass allows gradients to flow through the decay matrix correctly.
*   The validation of the recurrent formulation proves its viability for efficient $O(1)$ auto-regressive generation without KV-caching.
*   Next steps could involve scaling this component with Multi-Scale Retention (MSR) and integrating it into a full RetNet block structure.
