import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Gridworld environment for Dyna-Q
# 0 1 2 3
# 4 5 6 7
# 8 9 10 11 (11 is goal, 9 is pit)
n_states = 12
n_actions = 4
goal_state = 11
pit_state = 9

def step(state, action):
    # 0: up, 1: down, 2: left, 3: right
    row = state // 4
    col = state % 4

    if action == 0: row = max(0, row - 1)
    elif action == 1: row = min(2, row + 1)
    elif action == 2: col = max(0, col - 1)
    elif action == 3: col = min(3, col + 1)

    next_state = row * 4 + col

    if next_state == goal_state:
        return next_state, 10.0, True
    elif next_state == pit_state:
        return next_state, -10.0, True
    else:
        return next_state, -0.1, False

# Dyna-Q Parameters
alpha = 0.1
gamma = 0.95
epsilon = 0.1
planning_steps = 10
episodes = 100

Q = np.zeros((n_states, n_actions))
# Model: dictionary mapping (state, action) to (next_state, reward)
model = {}

for episode in range(episodes):
    state = 0 # Start state
    done = False
    while not done:
        # Epsilon-greedy action
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        # Real experience update
        best_next_a = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state, best_next_a] * (not done)
        Q[state, action] += alpha * (td_target - Q[state, action])

        # Update model
        model[(state, action)] = (next_state, reward, done)

        # Planning phase
        if len(model) > 0:
            # list(model.keys()) can be slow in a loop, but fine for small model
            keys = list(model.keys())
            for _ in range(planning_steps):
                idx = np.random.randint(len(keys))
                sim_state, sim_action = keys[idx]
                sim_next_state, sim_reward, sim_done = model[(sim_state, sim_action)]

                sim_best_next_a = np.argmax(Q[sim_next_state])
                sim_td_target = sim_reward + gamma * Q[sim_next_state, sim_best_next_a] * (not sim_done)
                Q[sim_state, sim_action] += alpha * (sim_td_target - Q[sim_state, sim_action])

        state = next_state

print(f"Final Q-value for start state (0): {np.max(Q[0]):.4f}")
print(f"Final Q-value near goal (7): {np.max(Q[7]):.4f}")
print("Dyna-Q execution successful.")
