"""
Artificial Bee Colony (ABC) Algorithm.

This script implements the Artificial Bee Colony algorithm, a swarm intelligence-based
optimization technique inspired by the foraging behavior of honey bees.
It minimizes the multidimensional Rosenbrock function.
"""

import numpy as np

def rosenbrock(x):
    """The Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

class ArtificialBeeColony:
    def __init__(self, obj_func, bounds, num_bees=40, max_iter=500, limit=100):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.num_bees = num_bees
        self.num_employed = num_bees // 2
        self.num_onlookers = num_bees - self.num_employed
        self.max_iter = max_iter
        self.limit = limit

        # Initialize population
        self.population = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.num_employed, self.dim)
        )
        self.fitness = np.zeros(self.num_employed)
        self.trials = np.zeros(self.num_employed)

        for i in range(self.num_employed):
            self.fitness[i] = self.evaluate_fitness(self.population[i])

        self.best_solution = self.population[np.argmax(self.fitness)].copy()
        self.best_fitness = np.max(self.fitness)

    def evaluate_fitness(self, solution):
        # We are minimizing, but ABC maximizes fitness.
        # So we use 1 / (1 + cost) for cost >= 0
        cost = self.obj_func(solution)
        if cost >= 0:
            return 1.0 / (1.0 + cost)
        else:
            return 1.0 + abs(cost)

    def employed_bees_phase(self):
        for i in range(self.num_employed):
            k = np.random.randint(self.num_employed)
            while k == i:
                k = np.random.randint(self.num_employed)

            j = np.random.randint(self.dim)
            phi = np.random.uniform(-1, 1)

            new_solution = self.population[i].copy()
            new_solution[j] = self.population[i, j] + phi * (self.population[i, j] - self.population[k, j])

            # Bound checking
            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

            new_fitness = self.evaluate_fitness(new_solution)

            if new_fitness > self.fitness[i]:
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def onlooker_bees_phase(self):
        prob = self.fitness / np.sum(self.fitness)

        t = 0
        i = 0
        while t < self.num_onlookers:
            if np.random.rand() < prob[i]:
                t += 1
                k = np.random.randint(self.num_employed)
                while k == i:
                    k = np.random.randint(self.num_employed)

                j = np.random.randint(self.dim)
                phi = np.random.uniform(-1, 1)

                new_solution = self.population[i].copy()
                new_solution[j] = self.population[i, j] + phi * (self.population[i, j] - self.population[k, j])

                # Bound checking
                new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

                new_fitness = self.evaluate_fitness(new_solution)

                if new_fitness > self.fitness[i]:
                    self.population[i] = new_solution
                    self.fitness[i] = new_fitness
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1
            i = (i + 1) % self.num_employed

    def scout_bees_phase(self):
        for i in range(self.num_employed):
            if self.trials[i] > self.limit:
                self.population[i] = np.random.uniform(
                    low=self.bounds[:, 0], high=self.bounds[:, 1], size=self.dim
                )
                self.fitness[i] = self.evaluate_fitness(self.population[i])
                self.trials[i] = 0

    def optimize(self):
        for iter in range(self.max_iter):
            self.employed_bees_phase()
            self.onlooker_bees_phase()
            self.scout_bees_phase()

            # Update best solution
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_solution = self.population[current_best_idx].copy()

            if (iter + 1) % 100 == 0:
                print(f"Iteration {iter+1}, Best Cost: {self.obj_func(self.best_solution):.6f}")

        return self.best_solution, self.obj_func(self.best_solution)


def main():
    print("Starting Artificial Bee Colony optimization for the 3D Rosenbrock function...")
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    abc = ArtificialBeeColony(rosenbrock, bounds, num_bees=50, max_iter=500, limit=50)
    best_x, best_cost = abc.optimize()

    print("\nOptimization completed.")
    print(f"Best solution found: {best_x}")
    print(f"Best cost: {best_cost:.6f}")

    assert best_cost < 1.0, f"Optimization failed to find a good minimum, cost {best_cost} >= 1.0"
    print("Artificial Bee Colony algorithm successfully found a valid minimum.")

if __name__ == "__main__":
    main()
