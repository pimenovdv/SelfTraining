import torch
import torch.nn as nn
import torch.optim as optim

class ModelWithInstanceNorm(nn.Module):
    def __init__(self):
        super(ModelWithInstanceNorm, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm1d(4)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    torch.manual_seed(42)
    model = ModelWithInstanceNorm()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    X = torch.randn(16, 1, 10)
    y = (X.sum(dim=(1,2)) > 0).long()

    print("Training Model with Instance Normalization...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print(f"Final Loss: {loss.item():.4f}")
    assert loss.item() < 0.5, "Model failed to converge."
    print("Success: Model with Instance Normalization successfully trained.")

if __name__ == "__main__":
    main()
