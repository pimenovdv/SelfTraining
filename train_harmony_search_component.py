import numpy as np

def rosenbrock(x):
    """Rosenbrock function, global minimum at (1,1,...) where f(x)=0"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def harmony_search(objective_func, bounds, hms=20, hmcr=0.9, par=0.3, bw=0.01, max_iter=1000):
    """
    Harmony Search algorithm for continuous optimization.
    hms: Harmony Memory Size
    hmcr: Harmony Memory Considering Rate
    par: Pitch Adjusting Rate
    bw: Bandwidth
    """
    n_vars = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    # Initialize Harmony Memory
    hm = np.random.uniform(lower_bounds, upper_bounds, (hms, n_vars))
    fitness = np.array([objective_func(x) for x in hm])

    # Sort memory
    sort_idx = np.argsort(fitness)
    hm = hm[sort_idx]
    fitness = fitness[sort_idx]

    for _ in range(max_iter):
        new_harmony = np.zeros(n_vars)
        for i in range(n_vars):
            if np.random.rand() < hmcr:
                # Choose from memory
                idx = np.random.randint(0, hms)
                new_harmony[i] = hm[idx, i]

                # Pitch adjustment
                if np.random.rand() < par:
                    adjustment = (np.random.rand() * 2 - 1) * bw
                    new_harmony[i] += adjustment
            else:
                # Random initialization
                new_harmony[i] = np.random.uniform(lower_bounds[i], upper_bounds[i])

            # Bound clipping
            new_harmony[i] = np.clip(new_harmony[i], lower_bounds[i], upper_bounds[i])

        new_fitness = objective_func(new_harmony)

        # Update memory if better than worst
        if new_fitness < fitness[-1]:
            hm[-1] = new_harmony
            fitness[-1] = new_fitness

            # Re-sort
            sort_idx = np.argsort(fitness)
            hm = hm[sort_idx]
            fitness = fitness[sort_idx]

    return hm[0], fitness[0]

if __name__ == "__main__":
    np.random.seed(42)
    bounds = [(-5.0, 5.0)] * 5
    best_x, best_fitness = harmony_search(rosenbrock, bounds, hms=30, hmcr=0.95, par=0.3, bw=0.05, max_iter=2000)
    print(f"Harmony Search - Best Fitness: {best_fitness:.6f}, Best Position: {best_x}")
