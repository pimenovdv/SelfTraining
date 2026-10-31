import numpy as np

def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def particle_swarm_optimization(cost_func, bounds, num_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    dim = len(bounds)
    bounds = np.array(bounds)

    # Initialize particles
    particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))

    # Initialize personal bests and global best
    pbest = particles.copy()
    pbest_scores = np.array([cost_func(p) for p in particles])

    gbest = pbest[np.argmin(pbest_scores)].copy()
    gbest_score = np.min(pbest_scores)

    for _ in range(max_iter):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (pbest[i] - particles[i]) +
                           c2 * r2 * (gbest - particles[i]))

            # Update position
            particles[i] = particles[i] + velocities[i]

            # Apply bounds
            particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])

            # Update bests
            score = cost_func(particles[i])
            if score < pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score
                if score < gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = score

    return gbest, gbest_score

if __name__ == "__main__":
    np.random.seed(42)
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    best_pos, best_score = particle_swarm_optimization(rosenbrock, bounds, num_particles=50, max_iter=500)
    print(f"Optimal parameters: {best_pos}")
    print(f"Final loss: {best_score:.6f}")
