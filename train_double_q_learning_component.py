import numpy as np

class DoubleQLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize two Q-tables
        self.q1 = np.zeros((num_states, num_actions))
        self.q2 = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        # Use average or sum of Q1 and Q2 to select the greedy action
        return np.argmax(self.q1[state] + self.q2[state])

    def update(self, state, action, reward, next_state):
        if np.random.rand() < 0.5:
            # Update Q1
            best_next_action = np.argmax(self.q1[next_state])
            td_target = reward + self.gamma * self.q2[next_state, best_next_action]
            td_error = td_target - self.q1[state, action]
            self.q1[state, action] += self.alpha * td_error
        else:
            # Update Q2
            best_next_action = np.argmax(self.q2[next_state])
            td_target = reward + self.gamma * self.q1[next_state, best_next_action]
            td_error = td_target - self.q2[state, action]
            self.q2[state, action] += self.alpha * td_error

def main():
    print("Testing Double Q-Learning Component...")

    num_states = 5
    num_actions = 2
    agent = DoubleQLearningAgent(num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1)

    # Simple deterministic gridworld: 0 -> 1 -> 2 -> 3 -> 4 (Goal: +1 reward)
    # Actions: 0 (Right), 1 (Left)

    np.random.seed(42)
    episodes = 500

    for _ in range(episodes):
        state = 0
        while state != 4:
            action = agent.choose_action(state)

            if action == 0:
                next_state = min(state + 1, 4)
            else:
                next_state = max(state - 1, 0)

            reward = 1.0 if next_state == 4 else 0.0

            agent.update(state, action, reward, next_state)
            state = next_state

    print("Learned Q1-values:")
    print(agent.q1)
    print("Learned Q2-values:")
    print(agent.q2)
    print("Average Q-values:")
    print((agent.q1 + agent.q2) / 2.0)

    # Check if action 0 (Right) is preferred over action 1 (Left) in all non-terminal states
    avg_q = (agent.q1 + agent.q2) / 2.0
    for s in range(4):
        if avg_q[s, 0] <= avg_q[s, 1]:
            print("Failure: Suboptimal policy learned.")
            return

    print("Success")

if __name__ == "__main__":
    main()
