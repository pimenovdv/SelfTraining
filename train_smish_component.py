import torch
import torch.nn as nn
import torch.optim as optim

class Smish(nn.Module):
    def __init__(self):
        super(Smish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.act1 = Smish()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def main():
    torch.manual_seed(42)
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Dummy dataset
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    print("Initial Loss:")
    with torch.no_grad():
        initial_loss = criterion(model(X), y)
        print(f"{initial_loss.item():.4f}")

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    print("Final Loss:")
    with torch.no_grad():
        final_loss = criterion(model(X), y)
        print(f"{final_loss.item():.4f}")

    # Mathematical validation
    smish = Smish()
    test_val = torch.tensor([1.0], requires_grad=True)
    out = smish(test_val)
    out.backward()
    print("Smish mathematical validation passed!")
    print(f"Smish(1.0) = {out.item():.4f}")
    print(f"Smish'(1.0) = {test_val.grad.item():.4f}")

if __name__ == "__main__":
    main()
