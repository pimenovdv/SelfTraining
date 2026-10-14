import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SwishGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwishGLU).
    SwishGLU(x) = Swish(x * W_1 + b_1) * (x * W_2 + b_2)
    where Swish(z) = z * sigmoid(beta * z).
    """
    def __init__(self, in_features, out_features, beta=1.0):
        super(SwishGLU, self).__init__()
        self.linear = nn.Linear(in_features, out_features * 2)
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        out = self.linear(x)
        out1, out2 = out.chunk(2, dim=-1)
        swish_out1 = out1 * torch.sigmoid(self.beta * out1)
        return swish_out1 * out2

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.swish_glu = SwishGLU(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.swish_glu(x)
        x = self.fc2(x)
        return x

def test_swish_glu():
    swish_glu = SwishGLU(5, 5)
    x = torch.randn(2, 5)
    y = swish_glu(x)
    print(f"SwishGLU output shape: {y.shape}")

def train_model():
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    print("Testing SwishGLU Component...")
    test_swish_glu()
    print("\nTraining Simple Model with SwishGLU...")
    train_model()
    print("\nSwishGLU Component successfully trained!")
