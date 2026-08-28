"""
Genetic Algorithm Component

This script implements and verifies a Genetic Algorithm (GA) component mathematically in pure NumPy.
The objective is to find the global minimum of the Rastrigin function, a non-convex function often used for testing optimization algorithms.
"""

import numpy as np

def rastrigin_function(x):
    """
    Rastrigin function for optimization testing.
    Global minimum is at x = [0, 0, ...] with f(x) = 0.
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def train_genetic_algorithm():
    np.random.seed(42)

    # Hyperparameters
    pop_size = 50          # Population size
    dim = 2                # Dimensionality of the problem
    num_generations = 200  # Number of generations
    mutation_rate = 0.15    # Mutation probability
    crossover_rate = 0.8   # Crossover probability
    bounds = [-5.12, 5.12] # Search space bounds

    # Initialize population randomly within bounds
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))

    best_fitness = -np.inf # We maximize -loss
    best_individual = None

    print("Starting Genetic Algorithm optimization...")
    print(f"Objective: Minimize Rastrigin function in {dim}D space.")

    for generation in range(num_generations):
        # 1. Evaluate fitness (we want to minimize the function, so fitness is negative value)
        fitness = np.array([-rastrigin_function(ind) for ind in population])

        # Track the best individual
        max_idx = np.argmax(fitness)
        if fitness[max_idx] > best_fitness:
            best_fitness = fitness[max_idx]
            best_individual = population[max_idx].copy()

        if (generation + 1) % 40 == 0 or generation == 0:
            print(f"Generation {generation+1:3d}: Best Loss = {-best_fitness:.6f}, Best Pos = {best_individual}")

        # 2. Selection (Tournament Selection)
        tournament_size = 3
        selected_indices = []
        for _ in range(pop_size):
            participants = np.random.choice(pop_size, tournament_size, replace=False)
            winner = participants[np.argmax(fitness[participants])]
            selected_indices.append(winner)

        parents = population[selected_indices]

        # 3. Crossover (Blend Crossover / Arithmetic Crossover)
        next_generation = []
        for i in range(0, pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[i+1] if i+1 < pop_size else parents[0]

            if np.random.rand() < crossover_rate:
                # Random blending factor for each dimension
                alpha = np.random.rand(dim)
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = alpha * parent2 + (1 - alpha) * parent1
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            next_generation.extend([child1, child2])

        next_generation = np.array(next_generation)[:pop_size]

        # 4. Mutation (Gaussian Mutation)
        # Apply mutation randomly based on mutation_rate
        mutation_mask = np.random.rand(pop_size, dim) < mutation_rate
        mutations = np.random.normal(0, 0.5, (pop_size, dim))
        next_generation += mutation_mask * mutations

        # 5. Boundary Check
        next_generation = np.clip(next_generation, bounds[0], bounds[1])

        # Replace old population
        population = next_generation

    print("-" * 30)
    print(f"Optimization completed.")
    print(f"Global Minimum is at [0.0, 0.0] with value 0.0")
    print(f"Found Minimum is at {best_individual} with value {-best_fitness:.6f}")

    # Assert successful optimization (close to 0)
    assert -best_fitness < 0.5, f"Genetic Algorithm failed to find a near-optimal solution. Best value: {-best_fitness}"
    print("Genetic Algorithm component verification: SUCCESS")

if __name__ == "__main__":
    train_genetic_algorithm()
