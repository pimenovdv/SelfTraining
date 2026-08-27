import numpy as np
import os
import argparse

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    # Fix dimensions for 2D inputs
    x = np.atleast_2d(x)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class A2CNetwork:
    def __init__(self, state_dim, hidden_dim, action_dim, lr=0.01):
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2. / state_dim)
        self.b1 = np.zeros(hidden_dim)

        self.W_actor = np.random.randn(hidden_dim, action_dim) * np.sqrt(2. / hidden_dim)
        self.b_actor = np.zeros(action_dim)

        self.W_critic = np.random.randn(hidden_dim, 1) * np.sqrt(2. / hidden_dim)
        self.b_critic = np.zeros(1)

        self.lr = lr

    def forward(self, state):
        state = np.atleast_2d(state)
        h1 = np.dot(state, self.W1) + self.b1
        a1 = relu(h1)
        logits = np.dot(a1, self.W_actor) + self.b_actor
        probs = softmax(logits)
        value = np.dot(a1, self.W_critic) + self.b_critic
        return probs, value, a1, h1

    def update(self, state, action, target_value, advantage):
        state = np.atleast_2d(state)

        # Forward pass
        probs, value, a1, h1 = self.forward(state)

        # Gradients for critic
        d_value = -2 * (target_value - value)
        d_W_critic = np.dot(a1.T, d_value)
        d_b_critic = np.sum(d_value, axis=0)

        # Gradients for actor
        d_logits = probs.copy()
        d_logits[0, action] -= 1
        d_logits *= advantage # Policy gradient theorem with baseline

        d_W_actor = np.dot(a1.T, d_logits)
        d_b_actor = np.sum(d_logits, axis=0)

        # Backprop to hidden layer
        d_a1 = np.dot(d_value, self.W_critic.T) + np.dot(d_logits, self.W_actor.T)
        d_h1 = d_a1 * relu_deriv(h1)

        d_W1 = np.dot(state.T, d_h1)
        d_b1 = np.sum(d_h1, axis=0)

        # Update weights
        self.W_critic -= self.lr * d_W_critic
        self.b_critic -= self.lr * d_b_critic
        self.W_actor -= self.lr * d_W_actor
        self.b_actor -= self.lr * d_b_actor
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

class SimpleEnv:
    def __init__(self):
        self.state = 0
    def reset(self):
        self.state = 0
        return np.array([self.state], dtype=float)
    def step(self, action):
        if action == 1:
            self.state += 1
        else:
            self.state -= 1

        reward = 1.0 if self.state == 5 else -0.1
        done = self.state == 5 or self.state == -5
        return np.array([self.state], dtype=float), reward, done

def train_a2c():
    env = SimpleEnv()
    model = A2CNetwork(1, 16, 2, lr=0.005)
    gamma = 0.99

    epochs = 500
    for epoch in range(epochs):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < 100:
            probs, value, _, _ = model.forward(state)

            # Add small epsilon for numerical stability in choice
            p = probs[0]
            p = p / p.sum()
            action = np.random.choice(2, p=p)

            next_state, reward, done = env.step(action)
            _, next_value, _, _ = model.forward(next_state)

            target_value = reward + (1 - done) * gamma * next_value[0, 0]
            advantage = target_value - value[0, 0]

            model.update(state, action, target_value, advantage)

            state = next_state
            total_reward += reward
            step_count += 1

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Total Reward: {total_reward:.4f}, Steps: {step_count}")

    print("Training completed.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A2C Component")
    args = parser.parse_args()

    print("Starting A2C training...")
    success = train_a2c()
    if success:
        print("A2C component trained successfully!")
    else:
        print("A2C component training failed.")
        exit(1)
