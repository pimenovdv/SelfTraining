import numpy as np
import os
import argparse
from collections import deque
import random

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class DuelingDQNNetwork:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2. / state_dim)
        self.b1 = np.zeros(hidden_dim)

        self.W_v = np.random.randn(hidden_dim, 1) * np.sqrt(2. / hidden_dim)
        self.b_v = np.zeros(1)

        self.W_a = np.random.randn(hidden_dim, action_dim) * np.sqrt(2. / hidden_dim)
        self.b_a = np.zeros(action_dim)

        self.action_dim = action_dim

    def forward(self, x):
        self.x = x
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = relu(self.z1)

        self.V = np.dot(self.a1, self.W_v) + self.b_v
        self.A = np.dot(self.a1, self.W_a) + self.b_a

        self.A_mean = np.mean(self.A, axis=1, keepdims=True)
        self.Q = self.V + (self.A - self.A_mean)
        return self.Q

    def backward(self, d_out):
        d_V = np.sum(d_out, axis=1, keepdims=True)

        d_A = d_out - np.mean(d_out, axis=1, keepdims=True)

        d_W_v = np.dot(self.a1.T, d_V)
        d_b_v = np.sum(d_V, axis=0)

        d_W_a = np.dot(self.a1.T, d_A)
        d_b_a = np.sum(d_A, axis=0)

        d_a1_V = np.dot(d_V, self.W_v.T)
        d_a1_A = np.dot(d_A, self.W_a.T)
        d_a1 = d_a1_V + d_a1_A

        d_z1 = d_a1 * relu_deriv(self.z1)

        d_W1 = np.dot(self.x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)

        return d_W1, d_b1, d_W_v, d_b_v, d_W_a, d_b_a

    def copy_weights_from(self, other):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W_v = other.W_v.copy()
        self.b_v = other.b_v.copy()
        self.W_a = other.W_a.copy()
        self.b_a = other.b_a.copy()

class SimpleEnv:
    def __init__(self):
        self.state = 2

    def step(self, action):
        if action == 0:
            self.state -= 1
        else:
            self.state += 1

        if self.state == 4:
            return self.state, 1.0, True
        elif self.state == 0:
            return self.state, -1.0, True
        else:
            return self.state, -0.1, False

    def reset(self):
        self.state = 2
        return self.state

def one_hot(state, size=5):
    vec = np.zeros((1, size))
    vec[0, state] = 1.0
    return vec

def main():
    parser = argparse.ArgumentParser(description="Train a Dueling DQN component.")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for replay buffer.")
    parser.add_argument("--buffer_size", type=int, default=1000, help="Replay buffer capacity.")
    parser.add_argument("--target_update", type=int, default=10, help="Update target network every N episodes.")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    q_network = DuelingDQNNetwork(state_dim=5, hidden_dim=args.hidden_dim, action_dim=2)
    target_network = DuelingDQNNetwork(state_dim=5, hidden_dim=args.hidden_dim, action_dim=2)
    target_network.copy_weights_from(q_network)

    env = SimpleEnv()
    replay_buffer = deque(maxlen=args.buffer_size)

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    print(f"Training Dueling DQN with hidden_dim={args.hidden_dim}, epochs={args.epochs}, lr={args.lr}, gamma={args.gamma}")

    for ep in range(args.epochs):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 20:
            x = one_hot(state)

            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                q_values = q_network.forward(x)
                action = np.argmax(q_values[0])

            next_state, reward, done = env.step(action)

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward
            steps += 1

            if len(replay_buffer) >= args.batch_size:
                batch = random.sample(replay_buffer, args.batch_size)

                states = np.vstack([one_hot(b[0]) for b in batch])
                actions = np.array([b[1] for b in batch])
                rewards = np.array([b[2] for b in batch])
                next_states = np.vstack([one_hot(b[3]) for b in batch])
                dones = np.array([b[4] for b in batch])

                q_values = q_network.forward(states)

                next_q_values = target_network.forward(next_states)
                max_next_q_values = np.max(next_q_values, axis=1)

                targets = rewards + args.gamma * max_next_q_values * (1 - dones)

                d_out = np.zeros_like(q_values)
                for i in range(args.batch_size):
                    d_out[i, actions[i]] = (q_values[i, actions[i]] - targets[i])

                d_out = 2.0 * d_out / args.batch_size

                dW1, db1, dW_v, db_v, dW_a, db_a = q_network.backward(d_out)

                q_network.W1 -= args.lr * dW1
                q_network.b1 -= args.lr * db1
                q_network.W_v -= args.lr * dW_v
                q_network.b_v -= args.lr * db_v
                q_network.W_a -= args.lr * dW_a
                q_network.b_a -= args.lr * db_a

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % args.target_update == 0:
            target_network.copy_weights_from(q_network)

        if ep % (args.epochs // 10) == 0 or ep == args.epochs - 1:
            print(f"Episode {ep}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

    state = env.reset()
    done = False
    print("\nEvaluating final policy:")
    success = False
    while not done:
        x = one_hot(state)
        q_values = q_network.forward(x)
        action = np.argmax(q_values[0])
        print(f"State: {state} -> Action: {action} (Q-values: {q_values[0]})")
        state, _, done = env.step(action)
        if state == 4:
            success = True

    assert success, "Failed to navigate to goal."

if __name__ == "__main__":
    main()
