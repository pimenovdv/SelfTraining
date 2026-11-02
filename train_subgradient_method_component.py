import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.linear(x)

def train_subgradient():
    torch.manual_seed(42)
    X = torch.randn(100, 5)
    true_w = torch.randn(5, 1)
    y = X @ true_w + 0.1 * torch.randn(100, 1)

    model = LinearRegression(5)
    # L1 Loss (Absolute Error) is non-differentiable at 0, making it a good candidate for subgradient method
    criterion = nn.L1Loss()

    # Using SGD with learning rate decay, which acts as a subgradient method when loss is non-smooth
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print("Starting Subgradient Method Training...")

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    print("Training Completed.")
    print(f"Final Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_subgradient()
