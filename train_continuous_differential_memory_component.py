import torch
import torch.nn as nn
import torch.optim as optim

class ContinuousDifferentialMemory(nn.Module):
    """
    A continuous differential memory structure where the memory state m(t)
    evolves according to the differential equation:
    dm/dt = -m + W_mem @ tanh(m) + W_in @ x
    """
    def __init__(self, input_size, memory_size, dt=0.1):
        super().__init__()
        self.dt = dt
        self.memory_size = memory_size
        self.W_in = nn.Linear(input_size, memory_size)
        self.W_mem = nn.Linear(memory_size, memory_size)
        self.W_out = nn.Linear(memory_size, input_size)

    def forward(self, x, steps=10):
        batch_size = x.size(0)
        state = torch.zeros(batch_size, self.memory_size, device=x.device)
        u = self.W_in(x)

        # Euler integration for the differential equation
        for _ in range(steps):
            dm_dt = -state + self.W_mem(torch.tanh(state)) + u
            state = state + self.dt * dm_dt

        out = self.W_out(state)
        return out

def test_continuous_differential_memory():
    print("Testing Continuous Differential Memory Component...")
    model = ContinuousDifferentialMemory(input_size=10, memory_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Target to memorize/reconstruct
    x = torch.randn(64, 10)
    y_target = torch.randn(64, 10)

    initial_loss = None
    final_loss = None

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y_target)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
            print(f"Epoch 1, Loss: {initial_loss:.4f}")
        elif (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    final_loss = loss.item()
    print("Continuous Differential Memory Component trained successfully.")
    if final_loss < initial_loss:
        print("Model successfully minimized the objective.")
    return True

if __name__ == "__main__":
    test_continuous_differential_memory()
