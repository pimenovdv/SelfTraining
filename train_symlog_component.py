import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

class SymLog(nn.Module):
    """
    Symmetric Logarithmic (SymLog) activation function.
    SymLog(x) = sign(x) * ln(|x| + 1)
    """
    def __init__(self):
        super(SymLog, self).__init__()

    def forward(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

class SimpleModel(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = SymLog()
        self.fc2 = nn.Linear(hidden_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def train_symlog_component():
    print("Starting SymLog Component Training...")

    torch.manual_seed(42)
    # Generate data with high variance to demonstrate SymLog's squashing ability
    X = torch.randn(100, 10) * 10
    y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(100, 1)

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

    os.makedirs("results", exist_ok=True)
    with open("results/symlog_results.json", "w") as f:
        f.write(json.dumps({"loss_history": loss_history}))

    print(f"Initial loss: {loss_history[0]:.4f}, Final loss: {loss_history[-1]:.4f}")
    assert loss_history[-1] < loss_history[0], "Model did not learn"
    print("SymLog Component Training Completed Successfully.")

if __name__ == "__main__":
    train_symlog_component()
