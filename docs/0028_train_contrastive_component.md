# Experiment 0028: Train Contrastive Learning (InfoNCE) Component

## Objective
To implement and train a Contrastive Learning model with a two-tower architecture using the InfoNCE loss in pure NumPy. This explores multimodal/multi-view representation alignment, cross-entropy over similarities, and temperature scaling, verifying the manual forward and backward passes.

## Setup
*   **Script:** `train_contrastive_component.py`
*   **Data:** Synthetic paired dataset representing two views of the same underlying concepts.
*   **Hyperparameters:** `input_dim` = 8, `hidden_dim` = 16, `out_dim` = 4, `tau` = 0.1, `epochs` = 5000, `learning_rate` = 0.01 (Adam)

## Execution
The training script was executed to verify the mathematical formulation of the two-tower model and the InfoNCE loss with temperature-scaled cosine similarities.

## Results
*   **Status:** Success.
*   **Initial Loss:** 8.8242
*   **Final Loss:** 0.1241
*   **Loss Reduction:** The model successfully minimized the InfoNCE loss, effectively aligning the representations of positive pairs while pushing apart negative pairs.

## Observations & Next Steps
*   The model successfully learned to map corresponding inputs from two different domains (views) into a shared representation space.
*   The temperature parameter $\tau$ was critical in scaling the logits to create informative gradients.
*   Manual backpropagation successfully routed the gradients from the cross-entropy over similarity matrix back through the L2 normalization and the respective towers.
