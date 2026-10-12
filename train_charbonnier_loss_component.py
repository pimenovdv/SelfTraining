import numpy as np

class CharbonnierLoss:
    def __init__(self, eps=1e-3):
        self.eps = eps

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(np.sqrt(np.square(y_pred - y_true) + self.eps**2))

    def backward(self):
        diff = self.y_pred - self.y_true
        grad = diff / np.sqrt(np.square(diff) + self.eps**2)
        return grad / self.y_pred.size

def test_charbonnier_loss():
    np.random.seed(42)
    # Mock data
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9], [3.2, 3.8]])

    loss_fn = CharbonnierLoss(eps=1e-3)
    loss = loss_fn.forward(y_pred, y_true)

    print(f"Initial loss: {loss:.4f}")

    # Gradient descent
    lr = 0.05
    for epoch in range(100):
        loss = loss_fn.forward(y_pred, y_true)
        grad = loss_fn.backward()
        y_pred = y_pred - lr * grad * y_pred.size

    final_loss = loss_fn.forward(y_pred, y_true)
    print(f"Final loss: {final_loss:.4f}")
    assert final_loss < 0.05, "Loss did not converge"
    print("Test passed.")

if __name__ == "__main__":
    test_charbonnier_loss()
