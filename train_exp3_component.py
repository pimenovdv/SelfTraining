import numpy as np

class EXP3:
    """
    EXP3 (Exponential-weight algorithm for Exploration and Exploitation)
    for adversarial multi-armed bandits.
    """
    def __init__(self, n_actions, gamma=0.1):
        self.n_actions = n_actions
        self.gamma = gamma
        self.weights = np.ones(n_actions)
        self.probabilities = np.ones(n_actions) / n_actions

    def select_action(self):
        # Calculate probabilities
        sum_weights = np.sum(self.weights)
        self.probabilities = (1.0 - self.gamma) * (self.weights / sum_weights) + (self.gamma / self.n_actions)

        # Sample action based on probabilities
        action = np.random.choice(self.n_actions, p=self.probabilities)
        return action

    def update(self, action, reward):
        # Estimated reward (only for the chosen action, 0 for others)
        estimated_reward = np.zeros(self.n_actions)
        estimated_reward[action] = reward / self.probabilities[action]

        # Update weights
        self.weights *= np.exp(self.gamma * estimated_reward / self.n_actions)
        # Prevent numerical overflow
        self.weights /= np.max(self.weights)

def evaluate_exp3():
    np.random.seed(42)
    n_actions = 5
    n_steps = 1000

    agent = EXP3(n_actions=n_actions, gamma=0.1)

    # Adversarial environment: best arm changes over time
    total_reward = 0

    for t in range(n_steps):
        action = agent.select_action()

        # Define reward distribution dynamically
        if t < n_steps // 2:
            # Arm 0 is optimal
            rewards = [0.8, 0.2, 0.1, 0.3, 0.4]
        else:
            # Environment changes: Arm 4 becomes optimal
            rewards = [0.1, 0.2, 0.1, 0.3, 0.9]

        reward = 1.0 if np.random.random() < rewards[action] else 0.0

        agent.update(action, reward)
        total_reward += reward

    print(f"EXP3 Total Reward over {n_steps} steps: {total_reward}")
    print(f"Final weights: {agent.weights}")
    print(f"Final probabilities: {agent.probabilities}")

    # Assert learning occurred
    # In the second half, arm 4 was optimal, so its probability should be high
    assert agent.probabilities[4] > 0.2, "EXP3 failed to adapt to the new optimal arm."
    print("EXP3 component verified successfully.")

if __name__ == "__main__":
    evaluate_exp3()
