import numpy as np

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def cem(objective_function, mean, std, n_iterations, pop_size, elite_frac):
    """
    Cross-Entropy Method for function minimization.
    """
    np.random.seed(42)
    n_elite = int(pop_size * elite_frac)

    best_x = None
    best_eval = float('inf')

    for i in range(n_iterations):
        population = np.random.normal(mean, std, size=(pop_size, len(mean)))
        evals = np.array([objective_function(x) for x in population])

        elite_indices = np.argsort(evals)[:n_elite]
        elites = population[elite_indices]

        mean = np.mean(elites, axis=0)
        std = np.std(elites, axis=0) + 1e-6

        if evals[elite_indices[0]] < best_eval:
            best_eval = evals[elite_indices[0]]
            best_x = elites[0]

        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}: Best Eval = {best_eval:.6f}, Mean = {mean}")

    return best_x, best_eval

if __name__ == "__main__":
    print("Starting Cross-Entropy Method training...")
    mean = np.array([0.0, 0.0])
    std = np.array([5.0, 5.0])
    n_iterations = 100
    pop_size = 1000
    elite_frac = 0.1

    best_x, best_eval = cem(rosenbrock, mean, std, n_iterations, pop_size, elite_frac)
    print(f"Final Best Point: {best_x}")
    print(f"Final Best Eval: {best_eval:.6f}")
    assert best_eval < 0.1, "CEM failed to converge near the global minimum."
    print("Training complete and successful.")
