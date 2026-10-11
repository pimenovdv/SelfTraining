import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json

class ACON(nn.Module):
    """
    Activate or Not (ACON) activation function.
    Proposed in 'Activate or Not: Learning Customized Activation'
    https://arxiv.org/abs/2009.04759
    """
    def __init__(self, in_features):
        super(ACON, self).__init__()
        self.p1 = nn.Parameter(torch.randn(1, in_features))
        self.p2 = nn.Parameter(torch.randn(1, in_features))
        self.beta = nn.Parameter(torch.ones(1, in_features))

    def forward(self, x):
        # ACON-C form: S(beta * (p1 * x - p2 * x)) * (p1 * x - p2 * x) + p2 * x
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

class SimpleModel(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACON(hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def train_acon_component():
    print("Starting ACON Component Training...")

    # Generate dummy data (e.g. regression task)
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.sum(X**2, dim=1, keepdim=True) + 0.1 * torch.randn(100, 1)

    model = SimpleModel(10, 20)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/acon_results.json", "w") as f:
        f.write(json.dumps({"loss_history": loss_history}))
    print("ACON Component Training Completed Successfully.")

if __name__ == "__main__":
    train_acon_component()
