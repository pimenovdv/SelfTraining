import torch
import torch.nn as nn
import torch.optim as optim

class HighwayLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.h = nn.Linear(size, size)
        self.t = nn.Linear(size, size)

    def forward(self, x):
        h_out = torch.relu(self.h(x))
        t_out = torch.sigmoid(self.t(x))
        return h_out * t_out + x * (1 - t_out)

class HighwayNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.highway_layers = nn.ModuleList([HighwayLayer(hidden_dim) for _ in range(num_layers)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_proj(x))
        for layer in self.highway_layers:
            x = layer(x)
        return self.output_proj(x)

def main():
    print("Starting Highway Network training...")
    torch.manual_seed(42)

    # Generate some dummy classification data
    X = torch.randn(1000, 20)
    # Simple non-linear relationship
    y = (X[:, 0] * X[:, 1] + X[:, 2] > 0).long()

    model = HighwayNetwork(input_dim=20, hidden_dim=32, num_layers=5, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y).float().mean()
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    preds = torch.argmax(model(X), dim=1)
    final_acc = (preds == y).float().mean()
    print(f"Final Accuracy: {final_acc:.4f}")
    assert final_acc > 0.8, "Model failed to learn"
    print("Training complete and verified.")

if __name__ == "__main__":
    main()
