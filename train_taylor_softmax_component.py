import numpy as np
import torch
import torch.nn as nn

def taylor_softmax(x, dim=-1):
    """
    Taylor Softmax uses the 2nd order Taylor expansion of the exponential function:
    exp(x) ~ 1 + x + 0.5 * x^2
    This polynomial is strictly positive for all real x, making it suitable for probabilities.
    """
    term = 1 + x + 0.5 * x**2
    return term / term.sum(dim=dim, keepdim=True)

class TaylorSoftmaxComponent(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return taylor_softmax(x, self.dim)

def test_component():
    print("Testing Taylor Softmax Component...")
    x = torch.tensor([[0.0, 1.0, 2.0],
                      [-1.0, -2.0, -3.0]])

    component = TaylorSoftmaxComponent(dim=-1)
    y = component(x)

    print("Input:")
    print(x)
    print("Output:")
    print(y)
    print("Sum along dim=-1:")
    print(y.sum(dim=-1))

    assert torch.allclose(y.sum(dim=-1), torch.ones_like(y.sum(dim=-1))), "Probabilities must sum to 1"
    assert torch.all(y > 0), "Probabilities must be strictly positive"
    print("Taylor Softmax tests passed.")

if __name__ == "__main__":
    test_component()
