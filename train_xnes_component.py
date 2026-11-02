import numpy as np
from scipy.linalg import expm

def rosenbrock(x):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

class xNES:
    def __init__(self, dim, obj_func, pop_size=None, learning_rate_mu=None, learning_rate_sigma=None, learning_rate_b=None):
        self.dim = dim
        self.obj_func = obj_func
        self.pop_size = pop_size if pop_size else 4 + int(3 * np.log(dim))

        # Learning rates
        self.learning_rate_mu = learning_rate_mu if learning_rate_mu else 1.0
        self.learning_rate_sigma = learning_rate_sigma if learning_rate_sigma else (9 + 3 * np.log(dim)) / (5 * dim * np.sqrt(dim))
        self.learning_rate_b = learning_rate_b if learning_rate_b else self.learning_rate_sigma

        self.mu = np.random.uniform(-2, 2, dim)
        self.sigma = 1.0
        self.B = np.eye(dim)  # B is scaled by sigma, so C = sigma^2 * B * B^T

        # Utilities
        u = np.maximum(0, np.log(self.pop_size / 2 + 1) - np.log(np.arange(1, self.pop_size + 1)))
        self.utilities = u / np.sum(u) - 1 / self.pop_size

    def step(self):
        # Sample population
        Z = np.random.randn(self.pop_size, self.dim)
        X = self.mu + self.sigma * np.dot(Z, self.B.T)

        # Evaluate fitness
        fitness = np.array([self.obj_func(x) for x in X])

        # Rank the population (lower is better for minimization)
        ranks = np.argsort(fitness)
        Z_ranked = Z[ranks]

        # Compute gradients
        grad_delta = np.dot(self.utilities, Z_ranked)
        grad_M = np.dot(Z_ranked.T, np.diag(self.utilities)).dot(Z_ranked) - np.sum(self.utilities) * np.eye(self.dim)

        grad_sigma = np.trace(grad_M) / self.dim
        grad_B = grad_M - grad_sigma * np.eye(self.dim)

        # Update parameters
        self.mu += self.learning_rate_mu * self.sigma * np.dot(self.B, grad_delta)
        self.sigma *= np.exp(self.learning_rate_sigma / 2 * grad_sigma)
        self.B = np.dot(self.B, expm(self.learning_rate_b / 2 * grad_B))

        return self.mu, self.sigma, fitness[ranks[0]]

    def optimize(self, max_iter=1000, tol=1e-6):
        best_fitness = float('inf')
        for i in range(max_iter):
            mu, sigma, current_best = self.step()
            if current_best < best_fitness:
                best_fitness = current_best

            if best_fitness < tol:
                print(f"Converged at iteration {i}")
                break

        return mu, best_fitness

if __name__ == "__main__":
    print("Testing Exponential Natural Evolution Strategies (xNES) component on Rosenbrock function...")
    xnes = xNES(dim=2, obj_func=rosenbrock, pop_size=50)
    best_solution, best_fitness = xnes.optimize(max_iter=2000)
    print(f"Optimal parameters: {best_solution}")
    print(f"Minimum value: {best_fitness}")
    print("xNES optimization successful!")
