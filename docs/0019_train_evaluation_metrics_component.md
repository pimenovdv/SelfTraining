# Experiment 0019: Train Evaluation Metrics Component

## Objective
To mathematically formulate, implement, and test core evaluation metrics used in language modeling and classification tasks: Softmax, Cross-Entropy Loss, Perplexity, and Accuracy. The goal is to verify their behavior during forward and backward passes using manual gradient calculations.

## Setup
*   **Script:** `train_evaluation_metrics_component.py`
*   **Data:** Synthetic random input vectors mapped to random target vocabulary indices.
*   **Hyperparameters:** `batch_size` = 32, `vocab_size` = 100, `epochs` = 1000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of the metrics and the combined Softmax-Cross Entropy backward pass.

## Results
*   **Status:** Success.
*   **Final Loss:** 0.0560
*   **Final Perplexity:** 1.0576
*   **Final Accuracy:** 1.0000
*   The model successfully learned to minimize the Cross-Entropy loss and Perplexity, while increasing Accuracy, demonstrating that the manual backward pass correctly guides the weights to predict the target classes.

## Observations & Next Steps
*   The combined gradient of Softmax and Cross-Entropy (`probs - targets`) is elegant and highly stable, avoiding the numerical issues that could arise if calculated separately.
*   Perplexity serves as an intuitive metric for evaluating language models, representing the exponentiated average negative log-likelihood.
*   This establishes the rigorous evaluation metrics required for Phase 1 of the AGI/ASI Roadmap.
