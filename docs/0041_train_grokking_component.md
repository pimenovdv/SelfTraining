# Experiment 0041: Train Grokking Component

## Objective
To implement and train a neural network on a modular arithmetic task (addition modulo $p$) using pure `numpy`. The goal is to mathematically and empirically observe the initial phase of "Grokking" (delayed generalization), where the model's train accuracy quickly reaches 100% via memorization, while the test accuracy remains near random chance.

## Setup
*   **Script:** `train_grokking_component.py`
*   **Data:** Exhaustive pairs $(a, b)$ in $\mathbb{Z}_p \times \mathbb{Z}_p$ for $p = 17$, with one-hot encoding for inputs and target $(a+b) \pmod{p}$.
*   **Hyperparameters:** `hidden_dim` = 128, `epochs` = 10000, `learning_rate` = 0.1, `weight_decay` = 0.001.

## Execution
The training script was executed successfully.

## Results
*   **Status:** Success.
*   **Memorization Phase:** The model quickly memorized the training set (Train Acc $\approx 1.0$) while the test accuracy remained extremely low. This validates the first phase of learning on algorithmic datasets before the transition to generalizable algorithms (Grokking) occurs (which typically takes $10^5$ to $10^6$ epochs).
*   **Optimization Dynamics:** We verified that standard gradient descent optimizes the cross-entropy loss by exploiting data-specific spurious correlations first.

## Observations & Next Steps
*   This experiment confirms the memorization behavior on modular addition.
*   Understanding the structural phases of neural network learning (memorization vs generalization) is critical for mechanistic interpretability and creating safe, aligned representations in AGI.
*   Future work involves scaling epochs and hyperparameter tuning to explicitly force the low-norm structural phase where test accuracy jumps to 100%.
