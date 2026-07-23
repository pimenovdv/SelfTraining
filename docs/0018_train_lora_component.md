# Experiment 0018: Train Low-Rank Adaptation (LoRA) Component

## Objective
To implement and train a Low-Rank Adaptation (LoRA) component. This component tests the hypothesis that freezing a pre-trained model weight matrix and injecting trainable rank-decomposition matrices can drastically reduce the number of trainable parameters for downstream tasks while performing competitively, using pure matrix operations and manual backpropagation.

## Setup
*   **Script:** `train_lora_component.py`
*   **Data:** Synthetic adaptation dataset (adapting base representations to new targets).
*   **Hyperparameters:** `rank (r)` = 2, `alpha` = 1.0, `epochs` = 5000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes for the LoRA adapter matrices, while the base weight matrix remains perfectly frozen.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error between the adapted predictions and the target values over 5000 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs, verifying that the low-rank adaptation successfully bridged the gap between the base predictions and the new downstream task.

## Observations & Next Steps
*   The LoRA implementation correctly demonstrates parameter-efficient fine-tuning principles.
*   Initializing matrix A with random noise and matrix B with zeros effectively ensured that the initial adapter state is identity (zero addition to base weights), which is theoretically sound.
*   Manual derivation of backpropagation for A and B validates that gradients only flow into these small matrices.
*   Next steps could involve integrating LoRA into the Attention mechanisms (Q, K, V projections) of the Transformer blocks to measure efficiency gains.
