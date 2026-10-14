import torch
import torch.nn as nn
import torch.optim as optim

class ArctanActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.arctan(x)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.act = ArctanActivation()
        self.linear2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

def test_arctan_activation():
    # Mathematical validation
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, requires_grad=True)
    act = ArctanActivation()
    y = act(x)
    print("Arctan Forward:", y.detach().numpy())

    y.sum().backward()
    print("Arctan Gradient (x):", x.grad.detach().numpy())

def train_model():
    torch.manual_seed(42)
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

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
    test_arctan_activation()
    train_model()
