import numpy as np
import torch
import torch.nn as nn

class TabuSearch:
    def __init__(self, objective_func, bounds, step_size=0.1, tabu_tenure=5, max_iter=100, num_neighbors=10):
        self.objective_func = objective_func
        self.bounds = bounds
        self.step_size = step_size
        self.tabu_tenure = tabu_tenure
        self.max_iter = max_iter
        self.num_neighbors = num_neighbors
        self.tabu_list = {}

    def get_neighbors(self, current_solution):
        neighbors = []
        dim = len(self.bounds)
        for _ in range(self.num_neighbors):
            perturbation = np.random.uniform(-self.step_size, self.step_size, dim)
            neighbor = current_solution + perturbation
            # Clip to bounds
            neighbor = np.clip(neighbor, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
            neighbors.append(neighbor)
        return neighbors

    def is_tabu(self, solution):
        for tabu_sol, tenure in self.tabu_list.items():
            if np.linalg.norm(np.array(tabu_sol) - solution) < self.step_size / 2:
                return True
        return False

    def update_tabu_list(self):
        to_remove = []
        for tabu_sol in self.tabu_list:
            self.tabu_list[tabu_sol] -= 1
            if self.tabu_list[tabu_sol] <= 0:
                to_remove.append(tabu_sol)
        for tabu_sol in to_remove:
            del self.tabu_list[tabu_sol]

    def optimize(self):
        dim = len(self.bounds)
        current_solution = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
        best_solution = current_solution.copy()
        best_fitness = self.objective_func(current_solution)

        for _ in range(self.max_iter):
            neighbors = self.get_neighbors(current_solution)

            best_neighbor = None
            best_neighbor_fitness = float('inf')

            for neighbor in neighbors:
                if not self.is_tabu(neighbor):
                    fitness = self.objective_func(neighbor)
                    if fitness < best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = fitness

            if best_neighbor is None:
                # All neighbors are tabu, just pick the best tabu neighbor as an aspiration criterion (simplified)
                for neighbor in neighbors:
                    fitness = self.objective_func(neighbor)
                    if fitness < best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = fitness

            current_solution = best_neighbor
            self.tabu_list[tuple(current_solution)] = self.tabu_tenure
            self.update_tabu_list()

            if best_neighbor_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = best_neighbor_fitness

        return best_solution, best_fitness

def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

def main():
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    optimizer = TabuSearch(objective_func=ackley_function, bounds=bounds, step_size=0.5, tabu_tenure=5, max_iter=200, num_neighbors=20)
    best_solution, best_fitness = optimizer.optimize()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

    if best_fitness < 5.0:
        print("Tabu Search optimization successful.")
    else:
        print("Tabu Search optimization failed.")

if __name__ == '__main__':
    main()
