import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Phish(nn.Module):
    """
    Phish Activation Function.
    f(x) = x * tanh(GELU(x))
    """
    def __init__(self):
        super(Phish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(nn.functional.gelu(x))

# Synthetic dataset (XOR problem)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Simple Feed Forward Network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.act1 = Phish()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def main():
    torch.manual_seed(42)
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    print("Training Phish Component...")
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}")

    # Test the model
    with torch.no_grad():
        test_outputs = model(X)
        print("\nFinal Predictions:")
        for i in range(len(X)):
            print(f"Input: {X[i].numpy()}, Target: {y[i].numpy()}, Prediction: {test_outputs[i].numpy()}")

if __name__ == "__main__":
    main()
