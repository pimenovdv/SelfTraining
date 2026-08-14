import numpy as np

def rosenbrock(x):
    """Rosenbrock function: f(x, y) = (a - x)^2 + b(y - x^2)^2, where a=1, b=100."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def simulated_annealing(objective_function, bounds, n_iterations, initial_temp, cooling_rate):
    """
    Simulated Annealing for function minimization.
    """
    np.random.seed(42)
    best_x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    best_eval = objective_function(best_x)
    curr_x = best_x
    curr_eval = best_eval
    temp = initial_temp

    for i in range(n_iterations):
        step = np.random.randn(len(bounds)) * 0.1 * (bounds[:, 1] - bounds[:, 0])
        candidate_x = curr_x + step
        candidate_x = np.clip(candidate_x, bounds[:, 0], bounds[:, 1])
        candidate_eval = objective_function(candidate_x)

        if candidate_eval < best_eval:
            best_x, best_eval = candidate_x, candidate_eval

        diff = candidate_eval - curr_eval
        if temp < 1e-8:
            temp = 1e-8

        with np.errstate(over='ignore'):
            metropolis = np.exp(-diff / temp)

        if diff < 0 or np.random.rand() < metropolis:
            curr_x, curr_eval = candidate_x, candidate_eval

        temp *= cooling_rate

        if (i+1) % 1000 == 0:
            print(f"Iteration {i+1}: Best Eval = {best_eval:.6f}, Temp = {temp:.6f}")

    return best_x, best_eval

if __name__ == "__main__":
    print("Starting Simulated Annealing training...")
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    n_iterations = 10000
    initial_temp = 10.0
    cooling_rate = 0.999

    best_x, best_eval = simulated_annealing(rosenbrock, bounds, n_iterations, initial_temp, cooling_rate)
    print(f"Final Best Point: {best_x}")
    print(f"Final Best Eval: {best_eval:.6f}")
    assert best_eval < 0.1, "Simulated Annealing failed to converge near the global minimum."
    print("Training complete and successful.")
