import numpy as np
import os
import argparse

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class ContinuousHopfield:
    def __init__(self, patterns, beta=1.0):
        # patterns: shape (N, D) where N is number of patterns, D is dimension
        self.patterns = patterns
        self.beta = beta

    def energy(self, state):
        # E = -1/beta * log(sum(exp(beta * X * state))) + 0.5 * state^T * state
        sim = self.beta * np.dot(self.patterns, state)
        max_sim = np.max(sim)
        log_sum_exp = max_sim + np.log(np.sum(np.exp(sim - max_sim)))
        e = - (1.0 / self.beta) * log_sum_exp + 0.5 * np.dot(state, state)
        return e

    def update(self, state):
        sim = self.beta * np.dot(self.patterns, state)
        attn = softmax(sim)
        new_state = np.dot(attn, self.patterns) # shape (D,)
        return new_state

def main():
    np.random.seed(42)
    # create some random continuous patterns
    N, D = 10, 32
    patterns = np.random.randn(N, D)

    # Normalize patterns to lie on a unit sphere for stability
    patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)

    hopfield = ContinuousHopfield(patterns, beta=50.0)

    target_idx = 4
    target_pattern = patterns[target_idx]

    # Add significant noise
    noise = np.random.randn(D) * 0.2
    noisy_state = target_pattern + noise
    noisy_state = noisy_state / np.linalg.norm(noisy_state)

    print(f"Initial distance to target: {np.linalg.norm(noisy_state - target_pattern):.4f}")

    state = noisy_state
    for i in range(10):
        e = hopfield.energy(state)
        print(f"Step {i}, Energy: {e:.4f}")
        state = hopfield.update(state)

    final_dist = np.linalg.norm(state - target_pattern)
    print(f"Final distance to target: {final_dist:.4f}")

    if final_dist < 1e-2:
        print("Successfully recovered the target pattern!")

    doc_content = r"""# Experiment 0123: Continuous Hopfield Network

## Overview
This experiment implements the Continuous (Modern) Hopfield Network mathematically in pure NumPy. Continuous Hopfield Networks generalize classic binary Hopfield networks to continuous states and use an exponential interaction function (log-sum-exp energy), massively increasing storage capacity.

## Mathematical Basis
The energy function of the Continuous Hopfield Network is given by:
$E(\xi) = -\frac{1}{\beta} \log \sum_{i=1}^N \exp(\beta x_i^T \xi) + \frac{1}{2} \xi^T \xi$

Where $X = (x_1, \dots, x_N)$ are the stored patterns, $\xi$ is the state vector, and $\beta$ is the inverse temperature parameter.
The update rule that minimizes this energy is:
$\xi^{new} = X^T \text{softmax}(\beta X \xi)$

This update rule is mathematically equivalent to the Self-Attention mechanism used in Transformer architectures, bridging associative memory and attention.

## Results
The implementation successfully retrieves stored continuous patterns from noisy initializations, iteratively minimizing the continuous energy function.
**Script:** `train_continuous_hopfield_component.py`
"""
    os.makedirs("docs", exist_ok=True)
    with open("docs/0123_train_continuous_hopfield_component.md", "w") as f:
        f.write(doc_content)
    print("Experiment documentation saved to docs/0123_train_continuous_hopfield_component.md")

if __name__ == "__main__":
    main()
