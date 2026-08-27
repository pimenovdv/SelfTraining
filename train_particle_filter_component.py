import numpy as np
import os
import argparse

def f_transition(x, process_noise):
    return x + np.sin(x) * 0.1 + process_noise

def h_observation(x, measurement_noise):
    return x**2 + measurement_noise

class ParticleFilter:
    def __init__(self, num_particles, process_variance, measurement_variance):
        self.num_particles = num_particles
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.particles = np.random.uniform(-5, 5, num_particles)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self):
        process_noise = np.random.normal(0, np.sqrt(self.process_variance), self.num_particles)
        self.particles = f_transition(self.particles, process_noise)

    def update(self, measurement):
        expected_measurement = h_observation(self.particles, 0)
        likelihood = np.exp(-0.5 * ((measurement - expected_measurement)**2) / self.measurement_variance)
        self.weights *= likelihood
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        indexes = np.searchsorted(cumulative_sum, np.random.random(self.num_particles))
        self.particles = self.particles[indexes]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, weights=self.weights)

def main():
    parser = argparse.ArgumentParser(description="Train Particle Filter Component.")
    parser.add_argument("--num_particles", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    np.random.seed(42)
    process_var = 1.0
    measurement_var = 1.0
    pf = ParticleFilter(args.num_particles, process_var, measurement_var)
    true_x = 2.0

    print("Testing Particle Filter (Sequential Monte Carlo)...")
    mses = []

    for t in range(args.steps):
        true_x = f_transition(true_x, np.random.normal(0, np.sqrt(process_var)))
        measurement = h_observation(true_x, np.random.normal(0, np.sqrt(measurement_var)))
        pf.predict()
        pf.update(measurement)
        estimate = pf.estimate()
        pf.resample()
        mse = (true_x - estimate)**2
        mses.append(mse)

        if (t+1) % 10 == 0:
            print(f"Step {t+1}: True X={true_x:.2f}, Est={estimate:.2f}, MSE={mse:.4f}")

    avg_mse = np.mean(mses)
    print(f"Average MSE: {avg_mse:.4f}")
    print("Particle Filter execution completed successfully.")

if __name__ == "__main__":
    main()
