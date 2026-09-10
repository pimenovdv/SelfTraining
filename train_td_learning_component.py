import numpy as np

def train():
    np.random.seed(42)
    # Simple Random Walk: States A, B, C, D, E.
    # Start at C. Ends at A (reward=0) or E (reward=1).
    # Actions: Left or Right (50/50).

    num_states = 7
    # 0: left terminal, 6: right terminal
    # 1:A, 2:B, 3:C, 4:D, 5:E

    V = np.zeros(num_states)

    # TD(0) learning
    alpha = 0.1
    gamma = 1.0
    episodes = 500

    for episode in range(episodes):
        state = 3 # Start at C

        while state != 0 and state != 6:
            # random policy
            action = np.random.choice([-1, 1])
            next_state = state + action

            reward = 1.0 if next_state == 6 else 0.0

            # TD update
            td_target = reward + gamma * V[next_state]
            td_error = td_target - V[state]
            V[state] += alpha * td_error

            state = next_state

    print("Estimated state values:")
    print(V[1:6])

    expected_V = np.array([1/6, 2/6, 3/6, 4/6, 5/6])
    print("Expected state values:")
    print(expected_V)

    mse = np.mean((V[1:6] - expected_V)**2)
    print(f"MSE: {mse:.4f}")
    assert mse < 0.01, "TD Learning failed to converge to true values."
    print("Success")

if __name__ == "__main__":
    train()
