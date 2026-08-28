import numpy as np

def rastrigin(x):
    """
    Rastrigin function for optimization testing.
    Global minimum is 0 at x = [0, ..., 0].
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def differential_evolution(objective_func, bounds, pop_size=20, mutate=0.5, recombination=0.7, max_iter=1000, tol=1e-6):
    """
    Differential Evolution algorithm implementation.
    """
    dimensions = len(bounds)
    # Initialize population
    pop = np.random.rand(pop_size, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop = min_b + pop * diff

    # Evaluate initial population
    fitness = np.asarray([objective_func(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]

    for i in range(max_iter):
        for j in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            mutant = np.clip(a + mutate * (b - c), min_b, max_b)

            cross_points = np.random.rand(dimensions) < recombination
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            f = objective_func(trial)

            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial

        if fitness[best_idx] < tol:
            break

    return best, fitness[best_idx]

def test_differential_evolution():
    print("Testing Differential Evolution component...")
    bounds = [(-5.12, 5.12)] * 5 # 5-dimensional Rastrigin
    best, best_fitness = differential_evolution(rastrigin, bounds, pop_size=30, mutate=0.6, recombination=0.8, max_iter=2000)

    print(f"Best solution found: {best}")
    print(f"Best fitness: {best_fitness:.6f}")

    assert best_fitness < 0.1, "Differential Evolution failed to converge to the global minimum."
    print("Differential Evolution component test passed successfully.")

if __name__ == "__main__":
    test_differential_evolution()
