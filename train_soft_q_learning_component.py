import numpy as np
import os

np.random.seed(42)

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        row = self.state // self.size
        col = self.state % self.size

        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.size - 1, col + 1)

        next_state = row * self.size + col
        self.state = next_state

        if next_state == self.size * self.size - 1:
            reward = 1.0
            done = True
        else:
            reward = -0.1
            done = False

        return next_state, reward, done

def main():
    print("Testing Soft Q-Learning Component...")
    env = GridWorld(size=4)

    num_states = 16
    num_actions = 4
    alpha = 0.1
    gamma = 0.99
    temperature = 1.0

    Q = np.zeros((num_states, num_actions))

    episodes = 5000
    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            # Softmax action selection
            q_vals = Q[state]
            exp_q = np.exp((q_vals - np.max(q_vals)) / temperature)
            probs = exp_q / np.sum(exp_q)
            action = np.random.choice(num_actions, p=probs)

            next_state, reward, done = env.step(action)

            # Soft value update
            next_q_vals = Q[next_state]
            max_next_q = np.max(next_q_vals)
            soft_v_next = max_next_q + temperature * np.log(np.sum(np.exp((next_q_vals - max_next_q) / temperature)))

            if done:
                target = reward
            else:
                target = reward + gamma * soft_v_next

            td_error = target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            steps += 1

        # Anneal temperature
        temperature = max(0.01, temperature * 0.999)

    # Test
    state = env.reset()
    path = [state]
    steps = 0
    done = False
    while not done and steps < 10:
        action = np.argmax(Q[state])
        state, _, done = env.step(action)
        path.append(state)
        steps += 1

    print(f"Path taken by greedy policy: {path}")
    if path[-1] == 15 and len(path) <= 8:
        print("Success! Soft Q-Learning converged to the optimal path.")
    else:
        print("Failed to find optimal path.")

if __name__ == "__main__":
    main()
