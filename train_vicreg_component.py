import numpy as np

class VICReg:
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, epsilon=1e-04):
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.epsilon = epsilon

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].reshape(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        N = z1.shape[0]
        D = z1.shape[1]

        # Invariance loss
        repr_loss = np.mean((z1 - z2) ** 2)

        # Variance loss
        std_z1 = np.sqrt(np.var(z1, axis=0) + self.epsilon)
        std_z2 = np.sqrt(np.var(z2, axis=0) + self.epsilon)
        std_loss = np.mean(np.maximum(0, 1 - std_z1)) + np.mean(np.maximum(0, 1 - std_z2))

        # Covariance loss
        z1_centered = z1 - np.mean(z1, axis=0)
        z2_centered = z2 - np.mean(z2, axis=0)

        cov_z1 = (z1_centered.T @ z1_centered) / (N - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (N - 1)

        cov_loss = (np.sum(self._off_diagonal(cov_z1) ** 2) + np.sum(self._off_diagonal(cov_z2) ** 2)) / D

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss, repr_loss, std_loss, cov_loss

def main():
    print("Testing VICReg Loss Component...")
    N, D = 128, 64

    vicreg = VICReg()

    np.random.seed(42)

    # Bad representations (collapsed to zero)
    z1_bad = np.zeros((N, D))
    z2_bad = np.zeros((N, D))

    loss_bad, rep_bad, std_bad, cov_bad = vicreg.forward(z1_bad, z2_bad)
    print(f"\nCollapsed representations (Zeros):")
    print(f"Total Loss: {loss_bad:.4f}, Repr Loss: {rep_bad:.4f}, Std Loss: {std_bad:.4f}, Cov Loss: {cov_bad:.4f}")

    # Good representations
    # Let's generate orthogonal features with variance > 1
    # Random orthogonal matrix
    H = np.random.randn(N, D)
    Q, _ = np.linalg.qr(H)

    # Q has orthonormal columns if N >= D. E[Q.T Q] = I
    # The variance of each column of Q is 1/N
    # We want variance of each column to be > 1. Let's say variance = 1.1
    # So we multiply by sqrt(1.1 * N)
    z1_good = Q * np.sqrt(1.1 * N)
    z2_good = z1_good + np.random.randn(N, D) * 0.01 # Small noise

    loss_good, rep_good, std_good, cov_good = vicreg.forward(z1_good, z2_good)
    print(f"\nGood representations (Decorrelated, High Variance):")
    print(f"Total Loss: {loss_good:.4f}, Repr Loss: {rep_good:.4f}, Std Loss: {std_good:.4f}, Cov Loss: {cov_good:.4f}")

    if loss_good < loss_bad:
        print("\nSuccess: Good representations have lower loss than collapsed representations.")
    else:
        print("\nFailure: Loss behavior is incorrect.")

if __name__ == "__main__":
    main()
