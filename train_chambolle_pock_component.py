import numpy as np

class ChambollePock:
    def __init__(self, tau=0.1, sigma=0.1, theta=1.0):
        self.tau = tau
        self.sigma = sigma
        self.theta = theta

    def optimize(self, x_init, K, K_adj, prox_F, prox_G_conj, n_iter=100):
        x = x_init.copy()
        x_bar = x_init.copy()
        y = np.zeros(K(x_init).shape)

        for _ in range(n_iter):
            # Dual update
            y = prox_F(y + self.sigma * K(x_bar), self.sigma)

            # Primal update
            x_old = x.copy()
            x = prox_G_conj(x - self.tau * K_adj(y), self.tau)

            # Extrapolation
            x_bar = x + self.theta * (x - x_old)

        return x

def test_chambolle_pock():
    # Simple example: minimize ||x - b||^2_2 + lambda ||x||_1
    np.random.seed(42)
    n = 10
    b = np.random.randn(n)
    lam = 0.5

    # Operators
    def K(x): return x
    def K_adj(y): return y

    # Proximal operator for F*(y) = (1/2)||y - b||^2
    # In convex conjugate space for (1/2)||Kx - b||^2, F*(y) = (1/2)||y||^2 + <y, b>
    def prox_F(y, sigma):
        return (y - sigma * b) / (1 + sigma)

    # Proximal operator for G(x) = lambda ||x||_1
    def prox_G(x, tau):
        return np.sign(x) * np.maximum(np.abs(x) - tau * lam, 0)

    cp = ChambollePock(tau=0.5, sigma=0.5, theta=1.0)
    x_init = np.zeros(n)

    x_opt = cp.optimize(x_init, K, K_adj, prox_F, prox_G, n_iter=200)
    print(f"Optimized x: {x_opt}")

    # Compare with soft thresholding directly on b
    x_true = np.sign(b) * np.maximum(np.abs(b) - lam, 0)
    print(f"True x: {x_true}")
    print(f"L2 error: {np.linalg.norm(x_opt - x_true):.4f}")
    assert np.linalg.norm(x_opt - x_true) < 1e-4

if __name__ == "__main__":
    print("Testing Chambolle-Pock algorithm...")
    test_chambolle_pock()
    print("Component test passed!")
