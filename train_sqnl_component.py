import torch
import torch.nn as nn
import os

class SQNL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(
            x > 2.0,
            torch.ones_like(x),
            torch.where(
                x >= 0.0,
                x - (x ** 2) / 4.0,
                torch.where(
                    x >= -2.0,
                    x + (x ** 2) / 4.0,
                    -torch.ones_like(x)
                )
            )
        )

def test_component():
    print("Testing SQNL Component...")
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    sqnl = SQNL()
    y = sqnl(x)
    print(f"Input: {x}")
    print(f"Output: {y}")

    # Simple training loop to verify it works
    x_train = torch.randn(100, 10)
    y_train = torch.randn(100, 10)
    model = nn.Sequential(
        nn.Linear(10, 10),
        SQNL(),
        nn.Linear(10, 10)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    initial_loss = criterion(model(x_train), y_train).item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x_train), y_train).item()
    print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Model did not learn"
    print("SQNL test passed.")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    test_component()
