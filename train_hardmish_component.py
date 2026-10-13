import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

class HardMish(nn.Module):
    def __init__(self):
        super(HardMish, self).__init__()

    def forward(self, x):
        return torch.where(x >= 0, x,
               torch.where(x >= -2, 0.5 * x**2 + x,
               torch.zeros_like(x)))

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act1 = HardMish()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def test_hardmish():
    # Test values
    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    hardmish = HardMish()
    y = hardmish(x)

    # Expected:
    # x = -3: 0
    # x = -2: 0.5 * (-2)^2 + (-2) = 2 - 2 = 0
    # x = -1: 0.5 * (-1)^2 + (-1) = 0.5 - 1 = -0.5
    # x = 0: 0
    # x = 1: 1
    # x = 2: 2

    expected = torch.tensor([0.0, 0.0, -0.5, 0.0, 1.0, 2.0])
    assert torch.allclose(y, expected, atol=1e-4)
    print("HardMish mathematical validation passed!")

def train_model():
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    X = torch.randn(100, 10)
    # y = sum(x) + noise
    Y = X.sum(dim=1, keepdim=True) + torch.randn(100, 1) * 0.1

    print("Starting training...")
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    if not os.path.exists("results"):
        os.makedirs("results")

    with open("results/hardmish_results.json", "w") as f:
        json.dump({"final_loss": float(loss.item())}, f)

    print("Training finished.")

if __name__ == "__main__":
    test_hardmish()
    train_model()
