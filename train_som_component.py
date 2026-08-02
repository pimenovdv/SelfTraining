import numpy as np
import os
import argparse

np.random.seed(42)

class SOM:
    def __init__(self, map_size, input_dim):
        self.map_size = map_size
        self.input_dim = input_dim
        self.W = np.random.randn(map_size[0] * map_size[1], input_dim)
        self.grid = np.indices(map_size).reshape(2, -1).T

    def train_step(self, x, lr, sigma):
        dist_sq = np.sum((self.W - x)**2, axis=1)
        bmu_idx = np.argmin(dist_sq)
        bmu_coord = self.grid[bmu_idx]
        grid_dist_sq = np.sum((self.grid - bmu_coord)**2, axis=1)
        influence = np.exp(-grid_dist_sq / (2 * (sigma**2) + 1e-8))
        self.W += lr * influence[:, np.newaxis] * (x - self.W)

def train_test():
    parser = argparse.ArgumentParser(description="Train a Self-Organizing Map (SOM).")
    args = parser.parse_args()

    centers = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    X = []
    for c in centers:
        X.append(np.random.randn(50, 2) * 0.1 + c)
    X = np.vstack(X)

    print("Training Self-Organizing Map (SOM)...")
    som = SOM(map_size=(10, 10), input_dim=2)

    epochs = 100
    n_samples = X.shape[0]
    initial_lr = 0.5
    initial_sigma = 5.0

    for epoch in range(epochs):
        np.random.shuffle(X)
        lr = initial_lr * np.exp(-epoch / epochs)
        sigma = initial_sigma * np.exp(-epoch / (epochs / np.log(initial_sigma)))
        for i in range(n_samples):
            som.train_step(X[i], lr, sigma)

    q_error = 0
    for i in range(n_samples):
        dist_sq = np.sum((som.W - X[i])**2, axis=1)
        q_error += np.sqrt(np.min(dist_sq))
    q_error /= n_samples

    print(f"Final Quantization Error: {q_error:.6f}")

    if q_error < 0.1:
        print("Success! Model learned topological mapping via SOM.")

        docs_dir = "docs"
        os.makedirs(docs_dir, exist_ok=True)
        report_path = os.path.join(docs_dir, "0073_train_som_component.md")

        report_content = f"""# 0073_train_som_component

## Status
Success

## Component
Self-Organizing Map (SOM)

## Description
Implemented and evaluated a Self-Organizing Map (SOM) component using pure NumPy. This component tests unsupervised learning of a lower-dimensional (2D grid) topological representation of higher-dimensional data, utilizing competitive learning and neighborhood influence functions.

## Results
- **Final Quantization Error:** {q_error:.6f}

The model successfully formed a topological map of the clustered input space.

**Script:** `train_som_component.py`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\nExperiment report saved to {report_path}")
    else:
        print("Failed.")

if __name__ == "__main__":
    train_test()
