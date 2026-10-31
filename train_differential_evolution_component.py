import numpy as np
import time

def rosenbrock(x):
    """The Rosenbrock function."""
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def differential_evolution(func, bounds, pop_size=50, mut=0.8, crossp=0.7, max_iter=1000):
    dimensions = len(bounds)
    pop = np.random.rand(pop_size, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([func(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(max_iter):
        for j in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = func(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

        # Add early stopping logic for convergence to known minimum of 0
        if fitness[best_idx] < 1e-8:
             break

    return best, fitness[best_idx]

def test_differential_evolution():
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    start_time = time.time()
    best_pos, best_val = differential_evolution(rosenbrock, bounds, max_iter=2000)
    end_time = time.time()

    print("Testing Differential Evolution Optimization Component...")
    print(f"Target: Minimize 3D Rosenbrock function")
    print(f"Global Minimum: [1. 1. 1.]")
    print(f"Discovered Minimum: {best_pos}")
    print(f"Final Value (Loss): {best_val:.6f}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")

    assert np.allclose(best_pos, np.ones(3), atol=1e-3), "Failed to converge near [1. 1. 1.]"
    assert best_val < 1e-4, "Failed to minimize function"
    print("Differential Evolution verification passed successfully!\n")

if __name__ == "__main__":
    test_differential_evolution()
