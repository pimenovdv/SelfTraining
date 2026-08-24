import numpy as np

class UCBMultiArmedBandit:
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_action(self):
        if self.total_counts < self.n_arms:
            return self.total_counts

        ucb_values = self.values + self.c * np.sqrt(np.log(self.total_counts) / self.counts)
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.counts[action] += 1
        self.total_counts += 1
        n = self.counts[action]
        value = self.values[action]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[action] = new_value

if __name__ == "__main__":
    n_arms = 3
    true_rewards = [0.1, 0.5, 0.8]
    agent = UCBMultiArmedBandit(n_arms=n_arms)

    np.random.seed(42)
    rewards = []
    for _ in range(1000):
        action = agent.select_action()
        reward = np.random.binomial(1, true_rewards[action])
        agent.update(action, reward)
        rewards.append(reward)

    print("Upper Confidence Bound component ran successfully.")
    print(f"Action counts: {agent.counts}")
    print(f"Estimated values: {agent.values}")
