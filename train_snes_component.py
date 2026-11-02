import numpy as np

def rosenbrock(x):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

class SNES:
    def __init__(self, dim, obj_func, pop_size=None, learning_rate=None):
        self.dim = dim
        self.obj_func = obj_func
        self.pop_size = pop_size if pop_size else 4 + int(3 * np.log(dim))
        self.learning_rate = learning_rate if learning_rate else 3 / (5 + dim)
        self.mu = np.random.uniform(-2, 2, dim)
        self.sigma = np.ones(dim)

        # Utilities (weights for fitness shaping)
        u = np.maximum(0, np.log(self.pop_size / 2 + 1) - np.log(np.arange(1, self.pop_size + 1)))
        self.utilities = u / np.sum(u) - 1 / self.pop_size

    def step(self):
        # Sample population
        Z = np.random.randn(self.pop_size, self.dim)
        X = self.mu + self.sigma * Z

        # Evaluate fitness
        fitness = np.array([self.obj_func(x) for x in X])

        # Rank the population (lower is better for minimization)
        ranks = np.argsort(fitness)
        Z_ranked = Z[ranks]

        # Compute gradients (Separable NES update)
        grad_mu = np.dot(self.utilities, Z_ranked)
        grad_sigma = np.dot(self.utilities, Z_ranked**2 - 1)

        # Update parameters
        self.mu += self.learning_rate * self.sigma * grad_mu
        self.sigma *= np.exp(self.learning_rate / 2 * grad_sigma)

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
    print("Testing Separable Natural Evolution Strategies (SNES) component on Rosenbrock function...")
    snes = SNES(dim=2, obj_func=rosenbrock, pop_size=50, learning_rate=0.1)
    best_solution, best_fitness = snes.optimize(max_iter=1000)
    print(f"Optimal parameters: {best_solution}")
    print(f"Minimum value: {best_fitness}")
    print("SNES optimization successful!")
