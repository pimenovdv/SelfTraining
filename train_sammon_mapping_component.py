"""
Sammon Mapping Component
This script evaluates Sammon Mapping mathematically in pure NumPy.
Sammon Mapping is a non-linear dimensionality reduction technique that aims to preserve
the structure of local neighborhoods by minimizing a stress function that penalizes
differences in pairwise distances, with a higher penalty for shorter distances.
"""

import numpy as np

class SammonMapping:
    def __init__(self, target_dim=2, max_iter=200, learning_rate=0.5, tol=1e-5):
        self.target_dim = target_dim
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol

    def fit_transform(self, X):
        N = X.shape[0]

        # Calculate original distance matrix
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(X[i] - X[j])
                D[i,j] = dist
                D[j,i] = dist

        # Initialize output points randomly
        Y = np.random.randn(N, self.target_dim) * 0.1

        # Calculate scaling factor c
        c = np.sum(D[np.triu_indices(N, k=1)])
        if c < 1e-10:
            return Y

        D_safe = np.copy(D)
        D_safe[D_safe < 1e-10] = 1e-10 # prevent div by zero

        prev_stress = float('inf')

        for iteration in range(self.max_iter):
            # Calculate current distance matrix in target space
            d = np.zeros((N, N))
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(Y[i] - Y[j])
                    d[i,j] = dist
                    d[j,i] = dist

            d_safe = np.copy(d)
            d_safe[d_safe < 1e-10] = 1e-10

            # Calculate gradient
            grad = np.zeros_like(Y)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        factor = (D_safe[i,j] - d[i,j]) / (D_safe[i,j] * d_safe[i,j])
                        grad[i] += factor * (Y[i] - Y[j])
            grad *= (-2.0 / c)

            # Update points
            Y -= self.learning_rate * grad

            # Calculate stress
            current_stress = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    current_stress += ((D_safe[i,j] - d[i,j])**2) / D_safe[i,j]
            current_stress /= c

            if abs(prev_stress - current_stress) < self.tol:
                break

            prev_stress = current_stress

        return Y, prev_stress

if __name__ == "__main__":
    print("Testing Sammon Mapping Component...")
    np.random.seed(42)

    # create synthetic data: 2 clusters in 3D
    cluster1 = np.random.randn(10, 3) + np.array([5, 5, 5])
    cluster2 = np.random.randn(10, 3) + np.array([-5, -5, -5])
    X = np.vstack((cluster1, cluster2))

    sm = SammonMapping(target_dim=2, max_iter=300, learning_rate=2.0)
    Y, final_stress = sm.fit_transform(X)

    print(f"Final Stress: {final_stress:.4f}")

    dist_intra = np.mean([np.linalg.norm(Y[i] - Y[j]) for i in range(10) for j in range(10) if i != j])
    dist_inter = np.mean([np.linalg.norm(Y[i] - Y[j]) for i in range(10) for j in range(10, 20)])

    print(f"Intra-cluster average distance: {dist_intra:.4f}")
    print(f"Inter-cluster average distance: {dist_inter:.4f}")

    if dist_inter > dist_intra:
        print("Success! Sammon Mapping successfully preserved local cluster structures in lower dimensions.")
    else:
        print("Failure. Clusters are not well separated.")
