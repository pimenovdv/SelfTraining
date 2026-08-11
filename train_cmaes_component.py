import numpy as np

class CMAES:
    def __init__(self, num_params, pop_size=10, initial_std=0.1):
        self.num_params = num_params
        self.pop_size = pop_size
        self.mean = np.zeros(num_params)
        self.std = initial_std
        self.cov = np.eye(num_params)
        self.weights = np.log(self.pop_size / 2 + 0.5) - np.log(np.arange(1, int(self.pop_size / 2) + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.cs = (self.mu_eff + 2) / (num_params + self.mu_eff + 5)
        self.cc = (4 + self.mu_eff / num_params) / (num_params + 4 + 2 * self.mu_eff / num_params)
        self.c1 = 2 / ((num_params + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((num_params + 2)**2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (num_params + 1)) - 1) + self.cs

        self.pc = np.zeros(num_params)
        self.ps = np.zeros(num_params)

    def ask(self):
        # Sample candidates
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.cov)
        self.eigenvalues = np.maximum(self.eigenvalues, 1e-8) # numerical stability
        self.D = np.diag(np.sqrt(self.eigenvalues))
        self.transform = np.dot(self.eigenvectors, self.D)

        self.z = np.random.randn(self.pop_size, self.num_params)
        candidates = self.mean + self.std * np.dot(self.z, self.transform.T)
        return candidates

    def tell(self, candidates, fitnesses):
        # Sort by fitness (minimize)
        sorted_indices = np.argsort(fitnesses)
        best_indices = sorted_indices[:len(self.weights)]
        best_candidates = candidates[best_indices]
        best_z = self.z[best_indices]

        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, None] * best_candidates, axis=0)

        # Evolution paths
        y = (self.mean - old_mean) / self.std
        z_mean = np.sum(self.weights[:, None] * best_z, axis=0)

        # Matrix C^-0.5
        inv_transform = np.dot(self.eigenvectors, np.diag(1 / np.sqrt(self.eigenvalues)))
        inv_transform = np.dot(inv_transform, self.eigenvectors.T)

        c_inv_y = np.dot(inv_transform, y)

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * c_inv_y

        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * 1)) / np.sqrt(self.num_params) < 1.4 + 2 / (self.num_params + 1)

        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y

        # Update Covariance
        artmp = (best_candidates - old_mean) / self.std
        self.cov = (1 - self.c1 - self.cmu) * self.cov \
                   + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.cov) \
                   + self.cmu * np.dot(artmp.T, np.diag(self.weights)).dot(artmp)

        # Update step size
        self.std = self.std * np.exp(self.cs / self.damps * (np.linalg.norm(self.ps) / np.sqrt(self.num_params) - 1))


def train_cmaes():
    np.random.seed(42)
    # Target function: Sphere function (minimize)
    def sphere(x):
        return np.sum(x**2)

    num_params = 10
    optimizer = CMAES(num_params=num_params, pop_size=20, initial_std=1.0)

    epochs = 100
    for epoch in range(epochs):
        candidates = optimizer.ask()
        fitnesses = np.array([sphere(c) for c in candidates])
        optimizer.tell(candidates, fitnesses)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Best Fitness: {np.min(fitnesses):.4f}")

    print("CMA-ES training component converged.")
    print("Final mean:", optimizer.mean)

if __name__ == "__main__":
    train_cmaes()
    print("train_cmaes_component.py successfully executed.")
