import torch
import torch.nn as nn
import torch.optim as optim

class Elliot(nn.Module):
    """
    Elliot activation function:
    f(x) = 0.5 * x / (1 + |x|) + 0.5
    Typically used scaled to [-1, 1], here we use standard formulation for activation scaling to [0, 1],
    or more precisely in [-1, 1]: f(x) = x / (1 + |x|). Let's use the latter.
    """
    def __init__(self):
        super(Elliot, self).__init__()

    def forward(self, x):
        return x / (1 + torch.abs(x))

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act1 = Elliot()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def test_elliot():
    # Test values
    x = torch.tensor([-1.0, 0.0, 1.0])
    elliot = Elliot()
    y = elliot(x)

    # Expected: x / (1 + |x|)
    # -1 / (1 + 1) = -0.5
    # 0 / (1 + 0) = 0.0
    # 1 / (1 + 1) = 0.5
    expected = torch.tensor([-0.5, 0.0, 0.5])
    assert torch.allclose(y, expected, atol=1e-4)
    print("Elliot mathematical validation passed!")

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
    test_elliot()
    train_model()
