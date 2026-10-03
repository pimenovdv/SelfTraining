import torch
import torch.nn as nn

class SmeLU(nn.Module):
    """
    Smooth ReLU (SmeLU) activation function.
    f(x) = 0 if x <= -beta
           (x + beta)^2 / (4 * beta) if -beta < x <= beta
           x if x > beta
    """
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return torch.where(
            x <= -self.beta,
            torch.zeros_like(x),
            torch.where(
                x >= self.beta,
                x,
                ((x + self.beta) ** 2) / (4 * self.beta)
            )
        )

def test_smelu():
    print("Testing SmeLU Activation Component...")
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    model = SmeLU(beta=2.0)
    out = model(x)
    expected = torch.tensor([0.0, 0.125, 0.5, 1.125, 3.0])
    print(f"Input: {x}")
    print(f"Output: {out}")
    print(f"Expected: {expected}")

    assert torch.allclose(out, expected), "Output doesn't match expected values"
    print("Test passed successfully!")

if __name__ == "__main__":
    test_smelu()
