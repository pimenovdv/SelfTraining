import torch
import torch.nn as nn

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups

        x = x.view(N, G, C // G, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(N, C, H, W)

        return x_norm * self.gamma + self.beta

def test_group_norm():
    print("Testing Group Normalization mathematically...")
    # N, C, H, W
    N, C, H, W = 2, 8, 4, 4
    num_groups = 4
    x = torch.randn(N, C, H, W)
    gn = GroupNorm(num_groups=num_groups, num_channels=C)
    y = gn(x)

    # Check mathematically per group
    y_grouped = y.view(N, num_groups, C // num_groups, H, W)
    y_mean = y_grouped.mean(dim=(2, 3, 4))
    y_var = y_grouped.var(dim=(2, 3, 4), unbiased=False)

    assert torch.allclose(y_mean, torch.zeros_like(y_mean), atol=1e-5), "Mean should be 0 per group"
    assert torch.allclose(y_var, torch.ones_like(y_var), atol=1e-5), "Variance should be 1 per group"
    print("Group Normalization verified successfully.")

if __name__ == "__main__":
    test_group_norm()
