import torch
import torch.nn as nn
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def test_layer_norm():
    print("Testing Layer Normalization mathematically...")
    x = torch.randn(32, 64)
    ln = LayerNorm(64)
    y = ln(x)

    # Check mathematically
    y_mean = y.mean(-1)
    y_std = y.std(-1, unbiased=False)

    assert torch.allclose(y_mean, torch.zeros_like(y_mean), atol=1e-5), "Mean should be 0"
    assert torch.allclose(y_std, torch.ones_like(y_std), atol=1e-5), "Std should be 1"
    print("Layer Normalization verified successfully.")

if __name__ == "__main__":
    test_layer_norm()
