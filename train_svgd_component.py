import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class SVGD:
    def __init__(self, target_score_fn, n_particles=100, step_size=0.1, n_iters=1000):
        self.target_score_fn = target_score_fn
        self.n_particles = n_particles
        self.step_size = step_size
        self.n_iters = n_iters

    def rbf_kernel(self, x):
        sq_dist = np.sum(x**2, axis=1, keepdims=True) + np.sum(x**2, axis=1) - 2 * np.dot(x, x.T)
        h = np.median(sq_dist)
        h = np.sqrt(0.5 * h / np.log(self.n_particles + 1))

        Kxy = np.exp(-sq_dist / h**2 / 2)
        dxkxy = -np.matmul(Kxy, x)
        sumkxy = np.sum(Kxy, axis=1, keepdims=True)
        dxkxy = (dxkxy + x * sumkxy) / (h**2)
        return Kxy, dxkxy

    def update(self, x):
        Kxy, dxkxy = self.rbf_kernel(x)
        score = self.target_score_fn(x)
        grad = (np.dot(Kxy, score) + dxkxy) / self.n_particles
        return x + self.step_size * grad

    def fit(self, x_init):
        x = np.copy(x_init)
        for i in range(self.n_iters):
            x = self.update(x)
        return x

def main():
    logging.info("Starting SVGD component training...")

    # Target distribution: 1D Gaussian N(mu=2, sigma=0.5)
    mu = 2.0
    sigma = 0.5

    def target_score(x):
        # Derivative of log PDF of N(mu, sigma)
        return -(x - mu) / (sigma**2)

    # Initialize particles randomly
    np.random.seed(42)
    x_init = np.random.normal(0, 1, (100, 1))

    svgd = SVGD(target_score_fn=target_score, n_particles=100, step_size=0.1, n_iters=500)
    x_final = svgd.fit(x_init)

    mean_est = np.mean(x_final)
    std_est = np.std(x_final)

    logging.info(f"Target Mean: {mu}, Estimated Mean: {mean_est:.4f}")
    logging.info(f"Target Std: {sigma}, Estimated Std: {std_est:.4f}")

    assert np.abs(mean_est - mu) < 0.1, "Mean estimation failed"
    assert np.abs(std_est - sigma) < 0.1, "Std estimation failed"

    logging.info("SVGD component successfully modeled the target distribution.")

if __name__ == "__main__":
    main()
