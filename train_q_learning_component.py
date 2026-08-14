"""
Tabular Q-Learning Component

This script implements tabular Q-Learning, a fundamental reinforcement learning
algorithm that learns an action-value function (Q-function) iteratively using
the Bellman equation, without requiring a model of the environment.
"""

import numpy as np

class SimpleGridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        self.num_states = size * size
        self.num_actions = 4 # up, right, down, left

    def reset(self):
        self.state = (0, 0)
        return self._state_to_idx(self.state)

    def _state_to_idx(self, state):
        return state[0] * self.size + state[1]

    def step(self, action):
        x, y = self.state
        if action == 0:   # up
            x = max(0, x - 1)
        elif action == 1: # right
            y = min(self.size - 1, y + 1)
        elif action == 2: # down
            x = min(self.size - 1, x + 1)
        elif action == 3: # left
            y = max(0, y - 1)

        self.state = (x, y)
        reward = 1.0 if self.state == self.goal else -0.01
        done = self.state == self.goal
        return self._state_to_idx(self.state), reward, done

class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

def main():
    print("Testing Tabular Q-Learning Component...")
    np.random.seed(42)

    env = SimpleGridWorld(size=4)
    agent = QLearningAgent(env.num_states, env.num_actions, epsilon=1.0)

    num_episodes = 500

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        # Decay epsilon
        agent.epsilon = max(0.1, agent.epsilon * 0.99)

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

    # Test greedy policy
    state = env.reset()
    done = False
    steps = 0
    agent.epsilon = 0.0 # pure greedy

    while not done and steps < 20:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        steps += 1

    print(f"Reached goal in {steps} steps (optimal is 6 for 4x4 grid).")
    assert done, "Agent failed to reach the goal using the learned policy"
    assert steps <= 6, "Agent did not learn the optimal path"
    print("Q-Learning test passed!")

if __name__ == "__main__":
    main()
