import numpy as np
import torch
import torch.nn as nn

class NaturalEvolutionStrategies:
    """
    Natural Evolution Strategies (NES).
    A black-box optimization algorithm that updates a search distribution
    (here, a multivariate Gaussian) in the direction of the natural gradient
    of expected fitness.

    This simple version uses a Gaussian search distribution with a mean vector `mu`
    and a scalar variance `sigma^2`.
    """
    def __init__(self, obj_func, bounds, pop_size=50, learning_rate_mu=0.1, learning_rate_sigma=0.01, sigma_init=1.0):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.lr_mu = learning_rate_mu
        self.lr_sigma = learning_rate_sigma

        # Initialize mean and standard deviation
        # Start mean at the center of the bounds
        self.mu = np.mean(self.bounds, axis=1)
        self.sigma = sigma_init

    def optimize(self, num_iterations=100):
        best_solution = None
        best_fitness = float('inf')

        for i in range(num_iterations):
            # Sample population from standard normal
            noise = np.random.randn(self.pop_size, self.dim)

            # Map to actual search space
            population = self.mu + self.sigma * noise

            # Clip population to bounds (optional, but good practice)
            for j in range(self.dim):
                population[:, j] = np.clip(population[:, j], self.bounds[j, 0], self.bounds[j, 1])

            # Evaluate fitness (minimization)
            fitness = np.array([self.obj_func(ind) for ind in population])

            # Update best solution found so far
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_solution = population[min_idx].copy()

            # Rank transformation or standardization of fitness (often used in NES for robustness)
            # Here we use standardized fitness (so lower fitness gets higher reward, since we minimize)
            # We want to maximize expected reward, so reward = -fitness
            rewards = -fitness
            rewards_standardized = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

            # Compute natural gradient updates
            # For a Gaussian with diagonal covariance (or scalar variance as here),
            # the natural gradient update simplifies nicely.

            # Gradient w.r.t mu
            grad_mu = np.dot(noise.T, rewards_standardized) / self.pop_size

            # Gradient w.r.t sigma (simplified for scalar sigma)
            # grad_sigma = E[reward * (noise^2 - 1) / sigma]
            grad_sigma = np.dot(np.sum(noise**2, axis=1) - self.dim, rewards_standardized) / self.pop_size

            # Update parameters
            self.mu += self.lr_mu * self.sigma * grad_mu
            self.sigma *= np.exp(self.lr_sigma / 2.0 * grad_sigma) # Multiplicative update to keep sigma positive

        return best_solution, best_fitness

def rastrigin(x):
    """Rastrigin function for continuous optimization testing."""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def main():
    print("Testing Natural Evolution Strategies (NES) component...")
    bounds = [(-5.12, 5.12)] * 2
    nes = NaturalEvolutionStrategies(rastrigin, bounds, pop_size=50, learning_rate_mu=0.1, learning_rate_sigma=0.01, sigma_init=2.0)
    best_x, best_f = nes.optimize(num_iterations=200)
    print(f"Best solution: {best_x}")
    print(f"Best fitness: {best_f:.4f}")
    assert best_f < 5.0, "NES failed to find a reasonable minimum for the Rastrigin function."
    print("NES component tested successfully.")

if __name__ == "__main__":
    main()
