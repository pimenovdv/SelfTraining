# Experiment 0114: Train Deep Deterministic Policy Gradient (DDPG) Component

## Objective
To implement and evaluate a Deep Deterministic Policy Gradient (DDPG) component mathematically. This tests the hypothesis that an actor-critic architecture can successfully learn deterministic policies in continuous action spaces by applying the deterministic policy gradient theorem, while utilizing target networks and experience replay for stability.

## Setup
*   **Script:** `train_ddpg_component.py`
*   **Data:** A simple continuous 1D environment (states -5.0 to 5.0), starting at state 0.0, goal at state >= 5.0.
*   **Hyperparameters:** `hidden_dim` = 64, `epochs` = 500, `lr_a` = 0.001, `lr_c` = 0.005, `gamma` = 0.99, `tau` = 0.005, `batch_size` = 64, `buffer_size` = 20000

## Execution
The training script was executed to verify the mathematical formulation of DDPG.
The critic learns to estimate the Q-value for state-action pairs by minimizing the TD error: $L(\theta^Q) = \mathbb{E} [(Q(s, a|\theta^Q) - y)^2]$ where $y = r + \gamma Q'(s', \mu'(s'|\theta^{\mu'})|\theta^{Q'})$.
The actor learns a deterministic policy by moving in the direction of the gradient of Q with respect to the action: $\nabla_{\theta^\mu} J \approx \mathbb{E} [\nabla_a Q(s, a|\theta^Q)|_{a=\mu(s)} \nabla_{\theta^\mu} \mu(s|\theta^\mu)]$.
Exploration is handled by adding decaying Gaussian noise to the deterministic action. Gradients were computed manually.

## Results
*   **Status:** Success.
*   **Learning:** The agent successfully learned to consistently output maximum positive actions to reach the goal state in continuous action space.
*   **Evaluation:** The final deterministic policy naturally drives the agent directly to the goal state.

## Observations & Next Steps
*   The implementation validates the core mechanics of DDPG, successfully utilizing the chain rule to backpropagate gradients from the critic to the actor.
*   Future work could explore TD3 (Twin Delayed DDPG) to address overestimation bias by using two critics and delayed policy updates.
