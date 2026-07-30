# Experiment 0061: Train REINFORCE (Policy Gradient) Component

## Objective
To implement and mathematically formulate the REINFORCE policy gradient algorithm. This tests the hypothesis that a neural network policy can learn to optimize expected returns in an environment by performing gradient ascent on the log probability of taken actions scaled by their respective rewards (with a baseline to reduce variance).

## Setup
*   **Script:** `train_reinforce_component.py`
*   **Data:** A simple 1D grid environment (states 0 to 4), starting at state 2, goal at state 4, pit at state 0.
*   **Hyperparameters:** `hidden_dim` = 16, `epochs` = 1000, `learning_rate` = 0.05, `gamma` = 0.99

## Execution
The training script was executed to verify the mathematical formulation of the policy gradient objective $J(\theta) = \mathbb{E}[\sum \gamma^t R_t \log \pi_\theta(a|s)]$ and its manual backpropagation (gradient ascent) with respect to the policy network weights.

## Results
*   **Status:** Success.
*   **Learning:** The agent successfully learned to navigate to the goal state (position 4) consistently.
*   **Evaluation:** The final policy deterministically selects the correct action (move right) with high probability from the starting states.

## Observations & Next Steps
*   The implementation validates the theoretical framework of policy gradients. By manually deriving the gradient of the log probability scaled by the standardized return, we confirm that the network correctly updates its weights to increase the likelihood of actions that lead to higher returns.
*   The use of a baseline (standardizing the returns in a batch) was crucial for reducing gradient variance and ensuring stable convergence, confirming a core principle of practical reinforcement learning.
*   Next steps could involve implementing Actor-Critic methods, combining this policy gradient approach with a value function baseline for even lower variance.
