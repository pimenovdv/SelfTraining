import numpy as np

class L1Loss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.abs(y_pred - y_true)
        if self.reduction == 'mean':
            return np.mean(loss)
        elif self.reduction == 'sum':
            return np.sum(loss)
        return loss

    def backward(self):
        grad = np.sign(self.y_pred - self.y_true)
        if self.reduction == 'mean':
            grad = grad / self.y_pred.size
        return grad

def test_component():
    np.random.seed(42)
    y_true = np.random.randn(10, 5)
    y_pred = np.random.randn(10, 5)

    criterion = L1Loss(reduction='mean')

    # Forward
    loss = criterion.forward(y_pred, y_true)
    print(f"L1 Loss (Mean): {loss:.4f}")

    # Backward
    grad = criterion.backward()
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient sum: {np.sum(grad):.4f}")
    print(f"Gradient mean: {np.mean(grad):.4f}")

    assert grad.shape == y_pred.shape, "Gradient shape mismatch"
    print("L1 Loss component tested successfully.")

if __name__ == "__main__":
    test_component()
