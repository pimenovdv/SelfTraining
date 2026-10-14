import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MishGLU(nn.Module):
    """
    Mish-Gated Linear Unit (MishGLU).
    MishGLU(x) = Mish(x * W_1 + b_1) * (x * W_2 + b_2)
    where Mish(z) = z * tanh(softplus(z)).
    """
    def __init__(self, in_features, out_features):
        super(MishGLU, self).__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        out = self.linear(x)
        out1, out2 = out.chunk(2, dim=-1)
        mish_out1 = out1 * torch.tanh(F.softplus(out1))
        return mish_out1 * out2

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.mishglu = MishGLU(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.mishglu(x)
        x = self.fc2(x)
        return x

def test_mishglu():
    mishglu = MishGLU(5, 5)
    x = torch.randn(2, 5)
    y = mishglu(x)
    print(f"MishGLU output shape: {y.shape}")

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
    print("Testing MishGLU Component...")
    test_mishglu()
    print("\nTraining Simple Model with MishGLU...")
    train_model()
    print("\nMishGLU Component successfully trained!")
