import torch
import torch.nn as nn
import torch.optim as optim

class MetaplasticityLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # Hidden variable dictating plasticity of each synapse (weight)
        self.plasticity = nn.Parameter(torch.ones(output_dim, input_dim))

    def forward(self, x):
        return self.linear(x)

def test_metaplasticity_component():
    model = MetaplasticityLayer(10, 5)
    x = torch.randn(2, 10)
    out = model(x)
    assert out.shape == (2, 5), "Output shape is incorrect"
    print("Metaplasticity layer test passed.")

if __name__ == "__main__":
    test_metaplasticity_component()
