import torch
import torch.nn as nn

class InstanceNorm(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x shape: (N, C, H, W) or (N, C, L)
        # Instance norm normalizes over spatial dimensions per channel per instance
        # Calculate mean and variance over all dimensions except N and C
        dims = tuple(range(2, x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Reshape gamma and beta to broadcast across spatial dimensions
        shape = [1, -1] + [1] * (x.dim() - 2)
        return self.gamma.view(*shape) * x_norm + self.beta.view(*shape)

def test_instance_norm():
    print("Testing Instance Normalization mathematically...")
    # Test on 2D inputs (e.g. N, C, H, W)
    N, C, H, W = 4, 3, 16, 16
    x = torch.randn(N, C, H, W)

    # Custom implementation
    in_norm = InstanceNorm(C)
    y_custom = in_norm(x)

    # Reference implementation
    ref_norm = nn.InstanceNorm2d(C, affine=True, eps=1e-5)

    # Check mathematically (mean should be ~0 and var should be ~1 per instance/channel)
    for i in range(N):
        for c in range(C):
            y_slice = y_custom[i, c]
            assert torch.allclose(y_slice.mean(), torch.tensor(0.0), atol=1e-4), f"Mean non-zero at instance {i}, channel {c}: {y_slice.mean()}"
            assert torch.allclose(y_slice.std(unbiased=False), torch.tensor(1.0), atol=1e-4), f"Std non-one at instance {i}, channel {c}: {y_slice.std()}"

    print("Instance Normalization verified successfully.")

if __name__ == "__main__":
    test_instance_norm()
