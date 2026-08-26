import numpy as np

def run_particle_filter():
    print("Testing Particle Filter (Sequential Monte Carlo)...")

    np.random.seed(42)
    T = 20
    Q = 10.0 # Process noise variance
    R = 1.0  # Observation noise variance

    x_true = np.zeros(T)
    y = np.zeros(T)

    x_true[0] = 0.1
    y[0] = (x_true[0]**2) / 20.0 + np.random.normal(0, np.sqrt(R))

    for t in range(1, T):
        x_true[t] = x_true[t-1] / 2 + 25 * x_true[t-1] / (1 + x_true[t-1]**2) + 8 * np.cos(1.2*(t-1)) + np.random.normal(0, np.sqrt(Q))
        y[t] = (x_true[t]**2) / 20.0 + np.random.normal(0, np.sqrt(R))

    num_particles = 1000
    particles = np.random.normal(0, np.sqrt(Q), num_particles)
    weights = np.ones(num_particles) / num_particles

    x_est = np.zeros(T)

    for t in range(T):
        if t > 0:
            particles = particles / 2 + 25 * particles / (1 + particles**2) + 8 * np.cos(1.2*(t-1)) + np.random.normal(0, np.sqrt(Q), num_particles)

        expected_y = (particles**2) / 20.0
        weights *= (1.0 / np.sqrt(2 * np.pi * R)) * np.exp(-0.5 * ((y[t] - expected_y)**2) / R)
        weights += 1e-300
        weights /= np.sum(weights)

        x_est[t] = np.sum(particles * weights)

        N_eff = 1.0 / np.sum(weights**2)
        if N_eff < num_particles / 2.0:
            positions = (np.random.random() + np.arange(num_particles)) / num_particles
            cumulative_sum = np.cumsum(weights)
            i, j = 0, 0
            new_particles = np.zeros(num_particles)
            while i < num_particles:
                if positions[i] < cumulative_sum[j]:
                    new_particles[i] = particles[j]
                    i += 1
                else:
                    j += 1
            particles = new_particles
            weights = np.ones(num_particles) / num_particles

    mse = np.mean((x_true - x_est)**2)
    print(f"Mean Squared Error: {mse:.4f}")

    if mse < 30.0:
        print("Particle Filter successfully tracked the non-linear system.")
        return True
    else:
        print("Particle Filter failed to track.")
        return False

if __name__ == "__main__":
    run_particle_filter()
