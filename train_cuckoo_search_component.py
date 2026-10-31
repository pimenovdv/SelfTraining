import numpy as np
import time
from scipy.special import gamma

def rosenbrock(x):
    """The Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def cuckoo_search(objective_function, bounds, n_nests, n_iterations, pa=0.25, beta=1.5):
    """
    Cuckoo Search algorithm for global optimization.
    """
    dim = len(bounds)

    # Initialize nests
    nests = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_nests, dim))
    fitness = np.apply_along_axis(objective_function, 1, nests)

    # Find initial best
    best_nest = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    # Pre-calculate sigma for Levy flight
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)

    for t in range(n_iterations):
        # Generate new solutions via Levy flights
        new_nests = np.copy(nests)

        # Calculate step for all nests
        u = np.random.normal(0, sigma, size=(n_nests, dim))
        v = np.random.normal(0, 1, size=(n_nests, dim))
        step = u / abs(v)**(1 / beta)

        step_size = 0.01 * step * (nests - best_nest)
        new_nests += step_size * np.random.randn(n_nests, dim)

        # Apply bounds
        new_nests = np.clip(new_nests, bounds[:, 0], bounds[:, 1])

        # Evaluate new solutions
        new_fitness = np.apply_along_axis(objective_function, 1, new_nests)

        # Update if better
        better_mask = new_fitness < fitness
        nests[better_mask] = new_nests[better_mask]
        fitness[better_mask] = new_fitness[better_mask]

        # Update best solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx]

        # Discovery of worse nests (fraction pa) and replacement
        random_nests1 = nests[np.random.randint(0, n_nests, n_nests)]
        random_nests2 = nests[np.random.randint(0, n_nests, n_nests)]
        step_size = np.random.rand(n_nests, 1) * (random_nests1 - random_nests2)

        replace_mask = np.random.rand(n_nests) < pa

        new_nests = np.copy(nests)
        new_nests[replace_mask] += step_size[replace_mask]
        new_nests = np.clip(new_nests, bounds[:, 0], bounds[:, 1])

        new_fitness = np.apply_along_axis(objective_function, 1, new_nests)

        # We take them if better
        better_mask = new_fitness < fitness
        nests[better_mask & replace_mask] = new_nests[better_mask & replace_mask]
        fitness[better_mask & replace_mask] = new_fitness[better_mask & replace_mask]

        # Update best solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx]

    return best_nest, best_fitness

def main():
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
    n_nests = 50
    n_iterations = 25000

    start_time = time.time()
    best_solution, best_fitness = cuckoo_search(rosenbrock, bounds, n_nests, n_iterations)
    execution_time = time.time() - start_time

    print(f"Optimization completed in {execution_time:.4f} seconds.")
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness (loss): {best_fitness:.6f}")

if __name__ == "__main__":
    main()
