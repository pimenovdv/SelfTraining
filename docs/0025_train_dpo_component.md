# Experiment 0025: Train Direct Preference Optimization (DPO) Component

## Objective
To implement and mathematically formulate Direct Preference Optimization (DPO). This tests the hypothesis that a language model policy can be directly aligned to human preferences by optimizing the log-ratio of policy to reference probabilities, completely bypassing the need for a separate reward model.

## Setup
*   **Script:** `train_dpo_component.py`
*   **Data:** Synthetic preference dataset consisting of 'chosen' and 'rejected' sequence pairs.
*   **Hyperparameters:** `d_model` = 4, `epochs` = 5000, `learning_rate` = 0.1, `beta` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of the DPO loss function and its manual backpropagation with respect to the policy weights.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The DPO loss successfully decreased over 5000 epochs.
*   **Predictions:** The final policy weights correctly shifted to assign higher implicit rewards to the 'chosen' sequences compared to the 'rejected' sequences, resulting in a preference probability > 0.5 for the chosen ones over the rejected ones.

## Observations & Next Steps
*   The implementation validates the theoretical framework of DPO. By formulating the reward implicitly via the log-ratio of the policy and reference models, we can optimize preferences directly using a simple binary cross-entropy objective.
*   Manual derivation of the gradients confirms that the policy weights are updated to increase the likelihood of the chosen sequence while decreasing the likelihood of the rejected sequence, scaled by the parameter `beta`.
*   Next steps could involve integrating DPO as a fine-tuning stage for the full Encoder-Decoder Transformer architecture.
