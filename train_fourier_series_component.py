import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

class FourierSeriesLayer(nn.Module):
    def __init__(self, in_features, num_harmonics=5):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.in_features = in_features
        self.weights_sin = nn.Parameter(torch.randn(num_harmonics, in_features))
        self.weights_cos = nn.Parameter(torch.randn(num_harmonics, in_features))
        self.bias = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        out = self.bias.unsqueeze(0).expand_as(x).clone()
        for n in range(1, self.num_harmonics + 1):
            out += self.weights_sin[n - 1] * torch.sin(n * x) + self.weights_cos[n - 1] * torch.cos(n * x)
        return out

def test_component():
    print("Testing Fourier Series Component...")

    model = FourierSeriesLayer(in_features=10, num_harmonics=3)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    inputs = torch.randn(16, 10)
    targets = torch.randn(16, 10)

    epochs = 100
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    print("Fourier Series component testing completed.")
    return {"loss_history": loss_history}

if __name__ == "__main__":
    results = test_component()
    os.makedirs("results", exist_ok=True)
    with open("results/fourier_series_results.json", "w") as f:
        json.dump(results, f)
