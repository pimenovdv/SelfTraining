import numpy as np

class SVRGOptimizer:
    def __init__(self, params, lr=0.01, update_freq=10):
        self.params = params
        self.lr = lr
        self.update_freq = update_freq
        self.snapshot_params = np.copy(params)
        self.full_grad = np.zeros_like(params)
        self.iter = 0

    def step(self, grad, grad_snapshot, full_grad):
        adjusted_grad = grad - grad_snapshot + full_grad
        self.params -= self.lr * adjusted_grad
        self.iter += 1

        if self.iter % self.update_freq == 0:
            self.snapshot_params = np.copy(self.params)
            return True
        return False

def test_svrg():
    np.random.seed(42)
    dim = 5
    Q = np.random.randn(dim, dim)
    Q = np.dot(Q.T, Q) + np.eye(dim) * 0.1
    c = np.random.randn(dim)

    params = np.random.randn(dim)
    opt = SVRGOptimizer(params, lr=0.01, update_freq=5)

    def compute_grad(x):
        return np.dot(Q, x) - c

    opt.full_grad = compute_grad(opt.snapshot_params)

    for i in range(50):
        noise = np.random.randn(dim) * 0.1
        grad = compute_grad(opt.params) + noise
        grad_snapshot = compute_grad(opt.snapshot_params) + noise

        update_full_grad = opt.step(grad, grad_snapshot, opt.full_grad)

        if update_full_grad:
            opt.full_grad = compute_grad(opt.snapshot_params)

        loss = 0.5 * np.dot(opt.params, np.dot(Q, opt.params)) - np.dot(c, opt.params)
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    print("SVRG component training completed successfully.")

if __name__ == "__main__":
    test_svrg()
