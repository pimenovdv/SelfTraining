# Experiment 0345: Train Trust Region Policy Optimization (TRPO) Component

## Objective
To implement and evaluate the Trust Region Policy Optimization (TRPO) algorithm mathematically. This tests the hypothesis that enforcing a KL-divergence constraint on policy updates ensures monotonic improvement and stabilizes learning compared to unconstrained policy gradients.

## Setup
*   **Script:** `train_trpo_component.py`
*   **Data:** A simple 1D grid environment (states 0 to 4), starting at state 2, goal at state 4, pit at state 0.
*   **Hyperparameters:** `hidden_dim` = 16, `epochs` = 2000, `lr_critic` = 0.01, `gamma` = 0.99, `max_kl` = 0.01

## Execution
The training script was executed to verify the mathematical formulation of TRPO. The actor updates its policy by maximizing the advantage using a natural gradient step. The natural gradient direction is found by solving $F p = \nabla_\theta J$ for $p$ using Conjugate Gradient, where $F$ is the Fisher Information Matrix approximated by a Fisher Vector Product. The step size is scaled to satisfy the KL constraint, and a backtracking line search ensures both constraint satisfaction and objective improvement.

## Results
*   **Status:** Success.
*   **Learning:** The agent successfully learned to navigate to the goal state consistently. The TRPO update constrained the policy changes, preventing catastrophic drops in performance.
*   **Evaluation:** The final policy robustly and deterministically selects the optimal action from starting states.

## Observations & Next Steps
*   The implementation validates the core mechanism of TRPO, successfully demonstrating conjugate gradient to find natural gradient steps and line search for constrained optimization in pure NumPy.
*   Future work could explore Proximal Policy Optimization (PPO), which achieves similar stability via a clipped surrogate objective instead of second-order optimization.
