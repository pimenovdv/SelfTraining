import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super(StarReLU, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale_value))
        self.bias = nn.Parameter(torch.tensor(bias_value))

    def forward(self, x):
        return self.scale * (torch.relu(x) ** 2) + self.bias

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act1 = StarReLU(0.8944, -0.4472) # Values often used for initialization
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def test_component():
    torch.manual_seed(42)
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    initial_loss = criterion(model(X), y).item()

    for _ in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(X), y).item()

    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")

    assert final_loss < initial_loss, "Model failed to learn"

    os.makedirs("results", exist_ok=True)
    with open("results/starrelu_results.json", "w") as f:
        json.dump({"initial_loss": initial_loss, "final_loss": final_loss}, f)

if __name__ == "__main__":
    test_component()
