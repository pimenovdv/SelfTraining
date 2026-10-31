import numpy as np
import time

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

class BatAlgorithm:
    def __init__(self, obj_func, num_bats, num_dims, bounds, max_iter):
        self.obj_func = obj_func
        self.num_bats = num_bats
        self.num_dims = num_dims
        self.bounds = bounds
        self.max_iter = max_iter

        self.f_min = 0.0
        self.f_max = 2.0

        self.A = np.ones(num_bats) * 0.5
        self.r = np.ones(num_bats) * 0.5

        self.Q = np.zeros(num_bats)
        self.v = np.zeros((num_bats, num_dims))

        self.x = np.random.uniform(bounds[0], bounds[1], (num_bats, num_dims))
        self.fitness = np.apply_along_axis(obj_func, 1, self.x)

        best_idx = np.argmin(self.fitness)
        self.best_x = self.x[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def optimize(self):
        alpha = 0.9
        gamma = 0.9

        for t in range(self.max_iter):
            for i in range(self.num_bats):
                self.Q[i] = self.f_min + (self.f_max - self.f_min) * np.random.uniform(0, 1)
                self.v[i] += (self.x[i] - self.best_x) * self.Q[i]
                x_new = self.x[i] + self.v[i]

                if np.random.rand() > self.r[i]:
                    x_new = self.best_x + 0.001 * np.random.randn(self.num_dims) * np.mean(self.A)

                x_new = np.clip(x_new, self.bounds[0], self.bounds[1])

                fitness_new = self.obj_func(x_new)

                if np.random.rand() < self.A[i] and fitness_new < self.fitness[i]:
                    self.x[i] = x_new
                    self.fitness[i] = fitness_new
                    self.A[i] *= alpha
                    self.r[i] = self.r[i] * (1 - np.exp(-gamma * t))

                if fitness_new < self.best_fitness:
                    self.best_x = x_new.copy()
                    self.best_fitness = fitness_new

        return self.best_x, self.best_fitness

def main():
    print("Training Bat Algorithm Component...")
    start_time = time.time()

    num_bats = 100
    num_dims = 3
    bounds = [-5, 5]
    max_iter = 50000

    ba = BatAlgorithm(rosenbrock, num_bats, num_dims, bounds, max_iter)
    best_pos, best_val = ba.optimize()

    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.4f} seconds.")
    print(f"Optimal parameters: {best_pos}")
    print(f"Optimal value: {best_val:.6f}")

if __name__ == '__main__':
    main()
