# Experiment 0062: Train Actor-Critic (RL) Component

## Objective
To implement and evaluate the Actor-Critic reinforcement learning algorithm mathematically. This tests the hypothesis that learning a Value function (Critic) simultaneously with a Policy (Actor) allows the use of Temporal Difference (TD) errors to reduce variance during gradient ascent, improving learning stability compared to standard REINFORCE.

## Setup
*   **Script:** `train_actor_critic_component.py`
*   **Data:** A simple 1D grid environment (states 0 to 4), starting at state 2, goal at state 4, pit at state 0.
*   **Hyperparameters:** `hidden_dim` = 16, `epochs` = 2000, `learning_rate` = 0.01, `gamma` = 0.99

## Execution
The training script was executed to verify the mathematical formulation of the Actor-Critic objective. The Actor updates using gradient ascent scaled by the TD error: $\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a|s) \delta$. The Critic minimizes the TD error loss: $\mathcal{L}_V = \frac{1}{2} \delta^2$, effectively optimizing its parameters via $\nabla_w \mathcal{L}_V = -\delta \nabla_w V_w(s)$. Both gradients were manually computed and applied via backpropagation through a shared hidden layer.

## Results
*   **Status:** Success.
*   **Learning:** The agent successfully learned to navigate to the goal state (position 4) consistently while learning value estimates for each state.
*   **Evaluation:** The final policy deterministically selects the correct action (move right) with high probability from the starting states, and the value estimates reflect the proximity to the reward.

## Observations & Next Steps
*   The implementation validates the theoretical framework of Actor-Critic methods. Utilizing a shared hidden layer correctly propagates gradients from both the actor's policy optimization and the critic's value estimation back through the network.
*   The use of TD errors enables step-by-step updates (online learning) without waiting for the end of an episode, setting a foundation for more advanced architectures like A2C and PPO.
