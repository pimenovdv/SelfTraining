import torch
import torch.nn as nn
import numpy as np

def rosenbrock(tensor):
    x, y = tensor[..., 0], tensor[..., 1]
    return (1 - x)**2 + 100 * (y - x**2)**2

class CEMOptimizer:
    def __init__(self, objective, dim=2, pop_size=100, elite_frac=0.1, max_iters=100, initial_std=1.0):
        self.objective = objective
        self.dim = dim
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.num_elite = max(1, int(self.pop_size * self.elite_frac))
        self.max_iters = max_iters
        self.mean = torch.zeros(dim)
        self.std = torch.ones(dim) * initial_std

    def optimize(self):
        best_overall_val = float('inf')
        best_overall_sol = None

        for i in range(self.max_iters):
            # Sample population
            population = self.mean + self.std * torch.randn(self.pop_size, self.dim)

            # Evaluate objective
            vals = self.objective(population)

            # Find elites
            elite_idx = torch.argsort(vals)[:self.num_elite]
            elites = population[elite_idx]

            # Update mean and std
            self.mean = elites.mean(dim=0)
            self.std = elites.std(dim=0) + 1e-5 # add small constant for numerical stability

            current_best_val = vals[elite_idx[0]].item()
            if current_best_val < best_overall_val:
                best_overall_val = current_best_val
                best_overall_sol = elites[0].clone()

            if (i+1) % 10 == 0:
                print(f"Iteration {i+1}: Best Eval = {best_overall_val:.6f}")

        return best_overall_sol, best_overall_val

if __name__ == '__main__':
    print("Starting Cross-Entropy Method training...")
    optimizer = CEMOptimizer(objective=rosenbrock)
    best_sol, best_val = optimizer.optimize()
    print(f"Final Best Point: {best_sol.numpy()}")
    print(f"Final Best Eval: {best_val:.6f}")
    print("Training complete and successful.")
