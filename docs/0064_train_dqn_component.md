# Experiment 0064: Train Deep Q-Network (DQN) Component

## Objective
To implement and evaluate a Deep Q-Network (DQN) mathematically. This tests the hypothesis that Q-learning can be stabilized using deep neural networks by introducing experience replay (to break temporal correlations) and a target network (to provide stable TD targets).

## Setup
*   **Script:** `train_dqn_component.py`
*   **Data:** A simple 1D grid environment (states 0 to 4), starting at state 2, goal at state 4, pit at state 0.
*   **Hyperparameters:** `hidden_dim` = 16, `epochs` = 1000, `learning_rate` = 0.01, `gamma` = 0.99, `batch_size` = 32, `buffer_size` = 1000

## Execution
The training script was executed to verify the mathematical formulation of DQN. The network learns to predict Q-values for actions, minimizing the Temporal Difference (TD) error: $L(\theta) = \mathbb{E}_{(s,a,r,s')} [ (r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta))^2 ]$. Gradients were computed manually, and experience replay was used to sample uncorrelated transitions.

## Results
*   **Status:** Success.
*   **Learning:** The agent successfully learned to navigate to the goal state consistently. The separation of the target network and the use of experience replay provided stable convergence.
*   **Evaluation:** The final policy deterministically selects the optimal action from starting states based on the highest Q-value.

## Observations & Next Steps
*   The implementation validates the core mechanisms of DQN. The stabilizing effects of the target network and replay buffer are mathematically sound and verifiable via manual backpropagation.
*   Future explorations could introduce Double DQN (to reduce overestimation bias) or Dueling DQN (to separate state-value and advantage estimation).
