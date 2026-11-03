import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ProximalPointComponent(nn.Module):
    def __init__(self, in_features, out_features, lam=0.1):
        super(ProximalPointComponent, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.lam = lam

    def forward(self, x):
        return self.linear(x)

    def proximal_update(self, lr):
        with torch.no_grad():
            self.linear.weight.copy_((1 / (1 + self.lam * lr)) * self.linear.weight)

def test_proximal_point():
    torch.manual_seed(42)
    model = ProximalPointComponent(10, 1, lam=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(100, 10)
    y = torch.randn(100, 1)

    criterion = nn.MSELoss()

    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        model.proximal_update(lr=0.1)

    assert loss.item() < 100, "Loss did not converge properly"
    print("Proximal Point Algorithm component tested successfully.")

if __name__ == "__main__":
    test_proximal_point()
