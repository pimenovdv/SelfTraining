import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SincActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Sinc(x) = sin(x) / x, and Sinc(0) = 1
        # To avoid division by zero, we use torch.where
        x_safe = torch.where(x == 0, torch.tensor(1e-20, device=x.device, dtype=x.dtype), x)
        return torch.where(x == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), torch.sin(x_safe) / x_safe)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.act = SincActivation()
        self.linear2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

def test_sinc_activation():
    # Mathematical validation
    x = torch.tensor([-3.14159, 0.0, 3.14159], dtype=torch.float32, requires_grad=True)
    act = SincActivation()
    y = act(x)
    print("Sinc Forward:", y.detach().numpy())

    y.sum().backward()
    print("Sinc Gradient:", x.grad.detach().numpy())

def train_sinc_model():
    torch.manual_seed(42)
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Dummy data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    print("Starting training...")
    for epoch in range(101):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    test_sinc_activation()
    train_sinc_model()
