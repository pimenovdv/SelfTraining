import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, num_particles, dimensions, bounds, w=0.5, c1=1.5, c2=1.5, seed=42):
        np.random.seed(seed)
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
        self.velocities = np.random.uniform(-1, 1, (num_particles, dimensions))

        self.pbest_positions = np.copy(self.positions)
        self.pbest_values = np.full(num_particles, np.inf)

        self.gbest_position = np.zeros(dimensions)
        self.gbest_value = np.inf

    def optimize(self, objective_func, num_iterations):
        for _ in range(num_iterations):
            for i in range(self.num_particles):
                fitness = objective_func(self.positions[i])

                if fitness < self.pbest_values[i]:
                    self.pbest_values[i] = fitness
                    self.pbest_positions[i] = self.positions[i]

                if fitness < self.gbest_value:
                    self.gbest_value = fitness
                    self.gbest_position = self.positions[i]

            for i in range(self.num_particles):
                r1 = np.random.rand(self.dimensions)
                r2 = np.random.rand(self.dimensions)

                cognitive_velocity = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.gbest_position - self.positions[i])

                self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

        return self.gbest_position, self.gbest_value

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

if __name__ == "__main__":
    print("Testing Particle Swarm Optimization (PSO) component...")

    dimensions = 2
    bounds = (-5.12, 5.12)
    pso = ParticleSwarmOptimization(num_particles=50, dimensions=dimensions, bounds=bounds, seed=42)

    best_pos, best_val = pso.optimize(rastrigin, num_iterations=100)

    print(f"Best Position: {best_pos}")
    print(f"Best Value (Fitness): {best_val:.6f}")

    if best_val < 1e-4:
        print("Successfully optimized the Rastrigin function to the global minimum!")
    else:
        print("Failed to find the global minimum.")
        exit(1)
