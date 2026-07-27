# Experiment 0044: Train Knowledge Distillation Component

## Objective
To implement and mathematically verify Knowledge Distillation (KD). This tests the hypothesis that a smaller "student" model can be trained to replicate the behavior of a larger "teacher" model by minimizing the KL Divergence between their temperature-scaled output probability distributions, in addition to the standard Cross-Entropy loss with hard labels.

## Setup
*   **Script:** `train_knowledge_distillation_component.py`
*   **Data:** Synthetic 2D classification dataset.
*   **Teacher Model:** MLP with hidden dimension = 16.
*   **Student Model:** MLP with hidden dimension = 4.
*   **Hyperparameters:** `epochs` = 5000, `learning_rate` = 0.1, `temperature (T)` = 3.0, `alpha (KD weight)` = 0.5

## Execution
The training script was executed to first train the teacher model to convergence. Then, the student model was trained using a combined loss function: `(1 - alpha) * Hard_CE_Loss + alpha * T^2 * Soft_KL_Loss`. Forward and backward passes were verified using manual `numpy` derivations.

## Results
*   **Status:** Success.
*   **Loss Reduction:** Both the teacher and student models successfully minimized their respective loss functions.
*   **Predictions:** The student model successfully learned the classification task, guided by both the hard labels and the softened probability distributions from the teacher. The gradient combination `d_logits = (1 - alpha) * d_logits_hard + alpha * d_logits_soft` worked correctly to update the student's weights.

## Observations & Next Steps
*   The implementation confirms that Knowledge Distillation can be effectively formulated and optimized mathematically without automatic differentiation.
*   Temperature scaling smooths the teacher's probability distribution, providing the student with richer information about the similarities between different classes (the "dark knowledge").
*   Next steps could involve distilling from a complex architecture (like a Transformer) to a simpler one (like an RNN or a smaller Transformer) for sequence modeling tasks.
