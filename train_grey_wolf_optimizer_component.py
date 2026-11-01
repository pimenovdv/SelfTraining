import numpy as np

def rosenbrock(x):
    """The Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def gwo(obj_func, dim, bounds, num_wolves=20, max_iter=100):
    """
    Grey Wolf Optimizer (GWO) algorithm.
    """
    # Initialize the positions of grey wolves
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")

    beta_pos = np.zeros(dim)
    beta_score = float("inf")

    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    # Initialize wolves randomly within bounds
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_wolves, dim))

    for l in range(max_iter):
        for i in range(num_wolves):
            # Return back the search agents that go beyond the boundaries of the search space
            positions[i, :] = np.clip(positions[i, :], bounds[:, 0], bounds[:, 1])

            # Calculate objective function for each search agent
            fitness = obj_func(positions[i, :])

            # Update Alpha, Beta, and Delta
            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        # a decreases linearly from 2 to 0
        a = 2 - l * (2 / max_iter)

        # Update the Position of search agents including omegas
        for i in range(num_wolves):
            for j in range(dim):
                # Alpha
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                # Beta
                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                # Delta
                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3

    return alpha_pos, alpha_score

if __name__ == "__main__":
    np.random.seed(42)

    dim = 5
    bounds = np.array([[-5.0, 5.0]] * dim)

    best_pos, best_score = gwo(rosenbrock, dim, bounds, num_wolves=50, max_iter=200)

    print(f"Grey Wolf Optimizer Test on Rosenbrock Function (dim={dim}):")
    print(f"Best Position: {best_pos}")
    print(f"Best Score: {best_score}")
