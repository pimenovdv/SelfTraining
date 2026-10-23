import numpy as np
import json
import os

def f(w):
    return -np.sum(w**2)

def train_es():
    np.random.seed(42)
    w = np.random.randn(5) * 2

    alpha = 0.05
    sigma = 0.1
    pop_size = 50

    print(f"Initial reward: {f(w):.4f}")

    for epoch in range(150):
        epsilon = np.random.randn(pop_size, len(w))
        rewards = np.zeros(pop_size)

        for i in range(pop_size):
            w_try = w + sigma * epsilon[i]
            rewards[i] = f(w_try)

        std_reward = np.std(rewards)
        if std_reward == 0:
            A = rewards - np.mean(rewards)
        else:
            A = (rewards - np.mean(rewards)) / std_reward

        w = w + alpha / (pop_size * sigma) * np.dot(epsilon.T, A)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, reward: {f(w):.4f}")

    final_reward = f(w)
    print(f"Final reward: {final_reward:.4f}")

    os.makedirs('results', exist_ok=True)
    with open('results/evolutionary_strategies_results.json', 'w') as f_out:
        json.dump({"final_reward": final_reward}, f_out)

if __name__ == "__main__":
    train_es()
