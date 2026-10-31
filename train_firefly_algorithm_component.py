import numpy as np
import time

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def firefly_algorithm(objective_fn, dim, pop_size=20, max_iters=200, alpha=0.5, beta_min=0.2, gamma=1.0):
    # Initialize fireflies
    np.random.seed(42)
    lower_bound = -5.0
    upper_bound = 5.0

    # Initialize positions and fitness
    positions = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    fitness = np.array([objective_fn(x) for x in positions])

    # Find initial best
    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx].copy()
    best_fitness = fitness[best_idx]

    for iteration in range(max_iters):
        for i in range(pop_size):
            for j in range(pop_size):
                if fitness[j] < fitness[i]:
                    # Distance between firefly i and j
                    r = np.linalg.norm(positions[i] - positions[j])

                    # Attractiveness
                    beta = (1.0 - beta_min) * np.exp(-gamma * r**2) + beta_min

                    # Movement
                    epsilon = np.random.uniform(-0.5, 0.5, dim)
                    positions[i] += beta * (positions[j] - positions[i]) + alpha * epsilon

                    # Apply bounds
                    positions[i] = np.clip(positions[i], lower_bound, upper_bound)

                    # Evaluate new fitness
                    fitness[i] = objective_fn(positions[i])

                    # Update best
                    if fitness[i] < best_fitness:
                        best_fitness = fitness[i]
                        best_pos = positions[i].copy()

        # Reduce alpha gradually
        alpha *= 0.97

    return best_pos, best_fitness

if __name__ == '__main__':
    start = time.time()
    best_x, best_loss = firefly_algorithm(rosenbrock, dim=3, pop_size=40, max_iters=300)
    end = time.time()

    print("Firefly Algorithm Component Test")
    print(f"Final Loss: {best_loss:.6f}")
    print(f"Execution Time: {end - start:.4f} seconds")
