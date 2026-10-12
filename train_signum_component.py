import numpy as np
import os

class Signum:
    """
    Signum Optimizer: SignSGD with Momentum.
    """
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in self.params]

    def step(self, grads):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay != 0:
                g = g + self.weight_decay * p

            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * g
            p -= self.lr * np.sign(self.m[i])

        return self.params

def test_component():
    print("Testing Signum Optimizer component...")

    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_w = np.array([1.5, -2.0, 3.0, -0.5, 0.8])
    true_b = 2.5
    y = X.dot(true_w) + true_b + np.random.randn(100) * 0.1

    w = np.zeros(5)
    b = np.zeros(1)

    optimizer = Signum([w, b], lr=0.1)

    epochs = 200
    initial_loss = None
    final_loss = None

    for epoch in range(epochs):
        y_pred = X.dot(w) + b
        loss = np.mean(np.square(y_pred - y))

        if epoch == 0:
            initial_loss = loss

        grad_y_pred = 2 * (y_pred - y) / len(y)
        grad_w = X.T.dot(grad_y_pred)
        grad_b = np.sum(grad_y_pred, keepdims=True)

        optimizer.step([grad_w, grad_b])

    final_loss = np.mean(np.square(X.dot(w) + b - y))

    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")

    assert final_loss < initial_loss, "Optimizer failed to reduce loss"
    assert final_loss < 0.5, "Optimizer did not converge well"

    print("Signum Optimizer test passed.")

if __name__ == "__main__":
    os.makedirs("models/signum", exist_ok=True)
    test_component()
