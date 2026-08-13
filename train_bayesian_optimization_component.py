import numpy as np
import math

# Set random seed for reproducibility
np.random.seed(42)

def objective_function(x):
    """
    Complex non-convex objective function.
    We want to maximize f(x) = x * np.sin(x)
    """
    return x * np.sin(x)

class GaussianProcess:
    def __init__(self, length_scale=1.0, noise_variance=1e-4):
        self.length_scale = length_scale
        self.noise_variance = noise_variance
        self.X = None
        self.y = None
        self.K_inv = None

    def rbf_kernel(self, X1, X2):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 / self.length_scale**2 * sqdist)

    def fit(self, X, y):
        self.X = X
        self.y = y
        K = self.rbf_kernel(self.X, self.X) + self.noise_variance * np.eye(len(self.X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_s):
        K_s = self.rbf_kernel(self.X, X_s)
        K_ss = self.rbf_kernel(X_s, X_s) + self.noise_variance * np.eye(len(X_s))

        mu_s = K_s.T.dot(self.K_inv).dot(self.y)
        cov_s = K_ss - K_s.T.dot(self.K_inv).dot(K_s)

        return mu_s.flatten(), np.diag(cov_s)

def norm_cdf(x):
    return 0.5 * (1 + np.array([math.erf(v / math.sqrt(2)) for v in x]))

def norm_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)

def expected_improvement(X_s, gp, y_best, xi=0.01):
    mu, sigma_sq = gp.predict(X_s)
    sigma = np.sqrt(np.maximum(sigma_sq, 1e-9))

    imp = mu - y_best - xi
    Z = np.zeros_like(sigma)

    mask = sigma > 0
    Z[mask] = imp[mask] / sigma[mask]

    ei = np.zeros_like(sigma)
    ei[mask] = imp[mask] * norm_cdf(Z[mask]) + sigma[mask] * norm_pdf(Z[mask])

    return ei

def bayesian_optimization(n_iters=15, n_init=3, bounds=(0, 10)):
    # Initialize with random points
    X_sample = np.random.uniform(bounds[0], bounds[1], size=(n_init, 1))
    Y_sample = objective_function(X_sample)

    gp = GaussianProcess(length_scale=1.5, noise_variance=1e-5)

    print("Starting Bayesian Optimization...")
    for i in range(n_iters):
        gp.fit(X_sample, Y_sample)

        # Grid search to optimize acquisition function (Expected Improvement)
        X_grid = np.linspace(bounds[0], bounds[1], 1000).reshape(-1, 1)
        y_best = np.max(Y_sample)

        ei = expected_improvement(X_grid, gp, y_best)

        # Find next point to evaluate
        next_idx = np.argmax(ei)
        next_x = X_grid[next_idx].reshape(1, 1)
        next_y = objective_function(next_x)

        X_sample = np.vstack((X_sample, next_x))
        Y_sample = np.vstack((Y_sample, next_y))

        print(f"Iteration {i+1}: Evaluated x = {next_x[0,0]:.4f}, f(x) = {next_y[0,0]:.4f}, Current Best f(x) = {np.max(Y_sample):.4f}")

    best_idx = np.argmax(Y_sample)
    return X_sample[best_idx, 0], Y_sample[best_idx, 0]

if __name__ == "__main__":
    best_x, best_y = bayesian_optimization(n_iters=15, n_init=5, bounds=(0, 10))
    print(f"\nOptimization Completed.")
    print(f"Best found: x = {best_x:.4f}, f(x) = {best_y:.4f}")

    # Global maximum of x * sin(x) in [0, 10] is around x=7.9787 with f(x)=7.9167
    assert best_y > 7.5, "Bayesian Optimization failed to find a near-optimal solution."
    print("Success: Bayesian Optimization effectively found a near-optimal point.")
