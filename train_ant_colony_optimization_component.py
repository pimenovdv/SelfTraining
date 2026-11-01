import numpy as np

def rosenbrock(x):
    """Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

class AntColonyOptimization:
    def __init__(self, obj_func, bounds, num_ants=20, max_iter=100, evaporation_rate=0.5, q_param=1.0, alpha=1.0, beta=2.0):
        self.obj_func = obj_func
        self.bounds = bounds
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.evaporation_rate = evaporation_rate
        self.q_param = q_param
        self.alpha = alpha
        self.beta = beta
        self.dim = len(bounds)
        self.best_pos = None
        self.best_cost = float('inf')

    def optimize(self):
        # We will adapt ACO for continuous domains.
        # This requires maintaining an archive of solutions and generating new ones based on Gaussian distributions around good solutions.
        # Initialize archive randomly within bounds
        archive_size = 50
        archive = np.zeros((archive_size, self.dim))
        costs = np.zeros(archive_size)

        for i in range(archive_size):
            for d in range(self.dim):
                archive[i, d] = np.random.uniform(self.bounds[d][0], self.bounds[d][1])
            costs[i] = self.obj_func(archive[i])

            if costs[i] < self.best_cost:
                self.best_cost = costs[i]
                self.best_pos = archive[i].copy()

        # Sort archive by cost
        sort_idx = np.argsort(costs)
        archive = archive[sort_idx]
        costs = costs[sort_idx]

        for iter in range(self.max_iter):
            # Probabilities of choosing an archive solution based on rank
            weights = np.zeros(archive_size)
            q = 0.1 # standard deviation parameter
            for i in range(archive_size):
                weights[i] = (1 / (q * archive_size * np.sqrt(2 * np.pi))) * np.exp(-((i**2) / (2 * (q**2) * (archive_size**2))))
            probs = weights / np.sum(weights)

            new_solutions = np.zeros((self.num_ants, self.dim))
            new_costs = np.zeros(self.num_ants)

            for ant in range(self.num_ants):
                # Choose a solution from archive to build around
                chosen_idx = np.random.choice(archive_size, p=probs)
                chosen_sol = archive[chosen_idx]

                # Standard deviation for Gaussian sampling based on distances to other solutions
                std_dev = np.zeros(self.dim)
                xi = 0.85 # Evaporation / convergence rate
                for d in range(self.dim):
                    diff = np.abs(archive[:, d] - chosen_sol[d])
                    std_dev[d] = xi * np.sum(diff) / (archive_size - 1)

                # Generate new solution
                for d in range(self.dim):
                    new_solutions[ant, d] = np.random.normal(chosen_sol[d], std_dev[d])
                    # Bound checking
                    new_solutions[ant, d] = np.clip(new_solutions[ant, d], self.bounds[d][0], self.bounds[d][1])

                new_costs[ant] = self.obj_func(new_solutions[ant])

            # Update archive: combine old and new, sort, keep best 'archive_size'
            combined_archive = np.vstack((archive, new_solutions))
            combined_costs = np.concatenate((costs, new_costs))

            sort_idx = np.argsort(combined_costs)
            archive = combined_archive[sort_idx][:archive_size]
            costs = combined_costs[sort_idx][:archive_size]

            if costs[0] < self.best_cost:
                self.best_cost = costs[0]
                self.best_pos = archive[0].copy()

            if (iter + 1) % 10 == 0:
                print(f"Iteration {iter + 1}/{self.max_iter}, Best Cost: {self.best_cost:.6f}")

        return self.best_pos, self.best_cost

if __name__ == "__main__":
    np.random.seed(42)
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    aco = AntColonyOptimization(rosenbrock, bounds, num_ants=30, max_iter=100)
    best_pos, best_cost = aco.optimize()
    print(f"\nOptimization Finished.")
    print(f"Best Position: {best_pos}")
    print(f"Best Cost: {best_cost:.6f}")
