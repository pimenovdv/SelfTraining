import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act1 = Mish()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def test_mish():
    # Test values
    x = torch.tensor([-1.0, 0.0, 1.0])
    mish = Mish()
    y = mish(x)

    # Expected: x * tanh(ln(1 + e^x))
    # softplus(-1) = ln(1 + e^-1) = ln(1 + 0.3678) = 0.3132, tanh(0.3132) = 0.3032, y = -0.3032
    # softplus(0) = ln(2) = 0.6931, tanh(0.6931) = 0.5999, y = 0
    # softplus(1) = ln(1 + e) = ln(3.718) = 1.3132, tanh(1.3132) = 0.8651, y = 0.8651

    expected = x * torch.tanh(F.softplus(x))
    assert torch.allclose(y, expected, atol=1e-4)
    print("Mish mathematical validation passed!")

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

    print("Training finished.")

if __name__ == "__main__":
    test_mish()
    train_model()
