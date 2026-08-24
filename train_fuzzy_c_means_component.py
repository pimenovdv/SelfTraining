import numpy as np

class FuzzyCMeans:
    def __init__(self, n_clusters, m=2.0, max_iters=100, tol=1e-4, seed=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.centers = None
        self.u = None

    def fit(self, X):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        self.u = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)

        for i in range(self.max_iters):
            u_old = self.u.copy()
            um = self.u ** self.m
            self.centers = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

            diff = X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            dist = np.fmax(dist, 1e-10)

            power = 2.0 / (self.m - 1)
            temp = dist ** power
            self.u = 1.0 / (temp * np.sum(1.0 / temp, axis=1, keepdims=True))

            if np.linalg.norm(self.u - u_old) < self.tol:
                break

if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    fcm = FuzzyCMeans(n_clusters=2)
    fcm.fit(X)
    print("Fuzzy C-Means component ran successfully.")
    print(f"Centers:\n{fcm.centers}")
