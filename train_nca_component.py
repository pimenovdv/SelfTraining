import numpy as np
import os
import argparse

np.random.seed(42)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class NeuralCellularAutomata:
    def __init__(self, grid_size, channels, hidden_dim):
        self.grid_size = grid_size
        self.channels = channels
        self.hidden_dim = hidden_dim

        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        self.W1 = np.random.randn(channels * 3, hidden_dim) * 0.1
        self.b1 = np.zeros((1, 1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, channels) * 0.01
        self.b2 = np.zeros((1, 1, channels))

    def perceive(self, x):
        h, w, c = x.shape
        perceived = np.zeros((h, w, c * 3))
        perceived[:, :, :c] = x

        for ch in range(c):
            padded = np.pad(x[:, :, ch], 1, mode='wrap')
            for i in range(h):
                for j in range(w):
                    patch = padded[i:i+3, j:j+3]
                    perceived[i, j, c + ch] = np.sum(patch * self.sobel_x)
                    perceived[i, j, 2*c + ch] = np.sum(patch * self.sobel_y)
        return perceived

    def forward(self, x, steps=5):
        self.history = [x.copy()]
        self.perceived_history = []
        self.h1_history = []
        self.z1_history = []

        for _ in range(steps):
            p = self.perceive(self.history[-1])
            self.perceived_history.append(p)

            z1 = np.dot(p, self.W1) + self.b1
            self.z1_history.append(z1)
            h1 = relu(z1)
            self.h1_history.append(h1)

            dx = np.dot(h1, self.W2) + self.b2

            mask = np.random.rand(self.grid_size, self.grid_size, 1) > 0.5

            next_x = self.history[-1] + dx * mask
            self.history.append(next_x)

        return self.history[-1]

    def backward(self, d_out, lr=0.01):
        dW1 = np.zeros_like(self.W1)
        dW2 = np.zeros_like(self.W2)

        steps = len(self.perceived_history)
        d_x = d_out

        for t in reversed(range(steps)):
            d_dx = d_x

            h1 = self.h1_history[t]
            p = self.perceived_history[t]
            z1 = self.z1_history[t]

            d_dx_flat = d_dx.reshape(-1, self.channels)
            h1_flat = h1.reshape(-1, self.hidden_dim)
            p_flat = p.reshape(-1, self.channels * 3)

            dW2 += np.dot(h1_flat.T, d_dx_flat)

            d_h1 = np.dot(d_dx_flat, self.W2.T).reshape(self.grid_size, self.grid_size, self.hidden_dim)
            d_z1 = d_h1 * relu_derivative(z1)
            d_z1_flat = d_z1.reshape(-1, self.hidden_dim)

            dW1 += np.dot(p_flat.T, d_z1_flat)

        self.W1 -= lr * dW1
        self.W2 -= lr * dW2


def train_test():
    grid_size = 5
    channels = 4
    model = NeuralCellularAutomata(grid_size, channels, 16)

    target = np.zeros((grid_size, grid_size, channels))
    target[1:4, 1:4, 0] = 1.0

    x0 = np.zeros((grid_size, grid_size, channels))
    x0[2, 2, 0] = 1.0

    print("Training Neural Cellular Automata (NCA)...")

    for epoch in range(1000):
        out = model.forward(x0, steps=5)
        loss = np.mean((out - target)**2)
        d_out = 2 * (out - target) / (grid_size * grid_size * channels)
        model.backward(d_out, lr=0.05)

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print(f"Final Loss: {loss:.6f}")

    if loss < 0.05:
        print("Success! Model learned target pattern growth via NCA.")

        docs_dir = "docs"
        os.makedirs(docs_dir, exist_ok=True)
        report_path = os.path.join(docs_dir, "0074_train_nca_component.md")

        report_content = f"""# 0074_train_nca_component

## Status
Success

## Component
Neural Cellular Automata (NCA)

## Description
Implemented and evaluated a Neural Cellular Automata (NCA) component using pure NumPy. This component tests the capacity of localized, iterative cell updates via a shared MLP and Sobel filters to learn to 'grow' a predefined target pattern from a single seed pixel. This tests self-organizing pattern generation.

## Results
- **Final Loss (MSE):** {loss:.6f}

The model successfully learned to iteratively grow the target square pattern from the seed.

**Script:** `train_nca_component.py`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\\nExperiment report saved to {report_path}")
    else:
        print("Failed.")

if __name__ == "__main__":
    train_test()
