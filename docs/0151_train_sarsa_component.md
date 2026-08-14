# SARSA Component

**Script:** `train_sarsa_component.py`

This experiment implements and tests State-Action-Reward-State-Action (SARSA), an on-policy temporal difference control algorithm, within a simple 4x4 GridWorld.

Unlike Q-learning, SARSA updates its Q-values using the action actually taken according to the epsilon-greedy policy. The agent successfully learned an optimal policy, demonstrating stable convergence during training and correctly navigating to the target state during testing without unnecessary steps.

- **Success:** Yes
- **Next Steps:** Evaluate in stochastic environments to contrast risk-averse vs risk-seeking behavior compared to Q-learning.
