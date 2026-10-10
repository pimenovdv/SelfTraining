import torch
import torch.nn as nn
import torch.optim as optim

class SphericalSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(SphericalSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Spherical softmax calculates probabilities proportional to squared values
        # after making them strictly positive (usually by squaring the inputs).
        # We square the inputs first: P_i = x_i^2 / sum(x_j^2)
        # To avoid division by zero, we add a small epsilon
        squared = x.pow(2)
        return squared / (squared.sum(dim=self.dim, keepdim=True) + 1e-8)

def test_component():
    print("Testing Spherical Softmax Component...")
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [-1.0, 0.0, 1.0]])

    component = SphericalSoftmax(dim=-1)
    y = component(x)

    print("Input:")
    print(x)
    print("Output:")
    print(y)
    print("Sum along dim=-1:")
    print(y.sum(dim=-1))

    assert torch.allclose(y.sum(dim=-1), torch.ones_like(y.sum(dim=-1))), "Probabilities must sum to 1"
    assert torch.all(y >= 0), "Probabilities must be non-negative"
    print("Spherical Softmax tests passed.")

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(5, 3)
        self.softmax = SphericalSoftmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)

def train_model():
    torch.manual_seed(42)
    model = SimpleNet()
    # Cross entropy expects logits, but since our activation outputs probabilities directly,
    # we'll use NLLLoss and apply log to the outputs of the model.
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Dummy data: Predict class 0 for negative sum, 1 for near zero, 2 for positive sum
    X = torch.randn(100, 5)
    sums = X.sum(dim=1)
    Y = torch.zeros(100, dtype=torch.long)
    Y[sums > 1] = 2
    Y[(sums >= -1) & (sums <= 1)] = 1
    Y[sums < -1] = 0

    print("Starting training...")
    for epoch in range(100):
        optimizer.zero_grad()
        probs = model(X)
        log_probs = torch.log(probs + 1e-8)
        loss = criterion(log_probs, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("Training finished.")

if __name__ == "__main__":
    test_component()
    train_model()
