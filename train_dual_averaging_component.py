import numpy as np

class DualAveraging:
    def __init__(self, n_features, learning_rate=1.0, gamma=1.0):
        self.w = np.zeros(n_features)
        self.z = np.zeros(n_features)
        self.t = 0
        self.lr = learning_rate
        self.gamma = gamma

    def step(self, grad):
        self.t += 1
        self.z += grad
        step_size = self.lr / (self.gamma * np.sqrt(self.t))
        self.w = -step_size * self.z
        return self.w

def test_dual_averaging():
    print("Testing Dual Averaging Component...")

    np.random.seed(42)
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([1.0, -0.5, 0.0, 2.0, -1.0])
    y = X.dot(true_w) + 0.1 * np.random.randn(n_samples)

    optimizer = DualAveraging(n_features, learning_rate=0.1, gamma=0.1)

    def compute_loss_and_grad(w):
        predictions = X.dot(w)
        errors = predictions - y
        loss = np.mean(errors ** 2)
        grad = 2 * X.T.dot(errors) / n_samples
        return loss, grad

    initial_loss, _ = compute_loss_and_grad(optimizer.w)
    print(f"Initial loss: {initial_loss:.4f}")

    epochs = 50
    for epoch in range(epochs):
        loss, grad = compute_loss_and_grad(optimizer.w)
        optimizer.step(grad)

    final_loss, _ = compute_loss_and_grad(optimizer.w)
    print(f"Final loss: {final_loss:.4f}")

    assert final_loss < initial_loss, "Dual Averaging failed to optimize the model"
    print("Optimization successful!")

if __name__ == "__main__":
    test_dual_averaging()
