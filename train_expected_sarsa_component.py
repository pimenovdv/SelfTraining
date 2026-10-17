import numpy as np

def train():
    np.random.seed(42)
    # Simple 1D Gridworld: 5 states (0 to 4)
    # Start at 2. Goal at 4 (reward 1). Pit at 0 (reward -1).
    # Actions: 0 (Left), 1 (Right)
    num_states = 5
    num_actions = 2

    Q = np.zeros((num_states, num_actions))

    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    episodes = 500

    for episode in range(episodes):
        state = 2 # Start state
        while state != 0 and state != 4:
            # Epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[state])

            # Take action
            next_state = state - 1 if action == 0 else state + 1

            # Get reward
            if next_state == 4:
                reward = 1.0
            elif next_state == 0:
                reward = -1.0
            else:
                reward = 0.0

            # Expected SARSA update
            if next_state == 0 or next_state == 4:
                expected_q = 0.0
            else:
                best_action = np.argmax(Q[next_state])
                expected_q = 0.0
                for a in range(num_actions):
                    prob = (1.0 - epsilon) + (epsilon / num_actions) if a == best_action else (epsilon / num_actions)
                    expected_q += prob * Q[next_state, a]

            td_target = reward + gamma * expected_q
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state

    print("Learned Q-values:")
    print(Q)

    # Validation
    assert Q[3, 1] > Q[3, 0], "Failed to learn correct action for state 3"
    assert Q[1, 1] > Q[1, 0], "Failed to learn correct action for state 1"

    print("Success")

if __name__ == "__main__":
    train()
