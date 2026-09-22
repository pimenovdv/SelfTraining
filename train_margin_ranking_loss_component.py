import numpy as np

class MarginRankingLoss:
    def __init__(self, margin=0.0):
        self.margin = margin

    def forward(self, x1, x2, y):
        """
        Computes Margin Ranking Loss.
        L = max(0, -y * (x1 - x2) + margin)
        """
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.diff = x1 - x2

        # Element-wise loss
        self.losses = np.maximum(0, -y * self.diff + self.margin)
        return np.mean(self.losses)

    def backward(self):
        """
        Computes gradients with respect to x1 and x2.
        """
        N = self.y.size
        # The gradient is non-zero only where the loss is > 0
        mask = self.losses > 0

        # dL/dx1 = -y / N if loss > 0 else 0
        grad_x1 = np.where(mask, -self.y / N, 0.0)

        # dL/dx2 = y / N if loss > 0 else 0
        grad_x2 = np.where(mask, self.y / N, 0.0)

        return grad_x1, grad_x2

def test_margin_ranking_loss_component():
    np.random.seed(42)

    # Generate some mock scores
    x1 = np.array([0.5, 0.1, 0.8, -0.2])
    x2 = np.array([0.2, 0.9, 0.5, 0.4])
    y = np.array([1, -1, 1, 1])  # 1 means x1 should be > x2, -1 means x2 should be > x1

    criterion = MarginRankingLoss(margin=0.1)

    # Forward pass
    loss = criterion.forward(x1, x2, y)
    print(f"Margin Ranking Loss: {loss:.4f}")

    # Backward pass
    grad_x1, grad_x2 = criterion.backward()

    print("Gradient wrt x1:", grad_x1)
    print("Gradient wrt x2:", grad_x2)

    # Optional: Numerical gradient check
    epsilon = 1e-5
    for i in range(len(x1)):
        x1_plus = x1.copy()
        x1_plus[i] += epsilon
        loss_plus = criterion.forward(x1_plus, x2, y)

        x1_minus = x1.copy()
        x1_minus[i] -= epsilon
        loss_minus = criterion.forward(x1_minus, x2, y)

        num_grad = (loss_plus - loss_minus) / (2 * epsilon)
        # Restore x1 state for criterion (implicitly done if we just re-run forward for original)
        criterion.forward(x1, x2, y)

        assert np.isclose(grad_x1[i], num_grad, atol=1e-4), f"Numerical gradient mismatch for x1[{i}]"

    print("Numerical gradient check passed for x1.")

    # Quick sanity check for PyTorch (optional reference)
    try:
        import torch
        import torch.nn as nn

        x1_t = torch.tensor(x1, requires_grad=True)
        x2_t = torch.tensor(x2, requires_grad=True)
        y_t = torch.tensor(y)

        criterion_t = nn.MarginRankingLoss(margin=0.1)
        loss_t = criterion_t(x1_t, x2_t, y_t)

        print(f"PyTorch Margin Ranking Loss: {loss_t.item():.4f}")
        assert np.isclose(loss, loss_t.item(), atol=1e-5), "Forward mismatch with PyTorch"

        loss_t.backward()
        assert np.allclose(grad_x1, x1_t.grad.numpy(), atol=1e-5), "x1 gradient mismatch with PyTorch"
        assert np.allclose(grad_x2, x2_t.grad.numpy(), atol=1e-5), "x2 gradient mismatch with PyTorch"
        print("PyTorch consistency check passed.")
    except ImportError:
        print("PyTorch not installed, skipping consistency check.")

if __name__ == "__main__":
    test_margin_ranking_loss_component()
