import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.gate(x) # (batch_size, num_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # (batch_size, num_experts, output_dim)
        output = torch.einsum('be,beo->bo', weights, expert_outputs)
        return output

def test_component():
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic data for a non-linear task
    X = np.random.randn(200, 10).astype(np.float32)
    y = (np.sin(X[:, 0]) + X[:, 1]**2 > 0.5).astype(np.float32).reshape(-1, 1)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    model = MoE(input_dim=10, hidden_dim=16, output_dim=1, num_experts=4)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training Mixture of Experts Component...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

    print("Training completed.")

if __name__ == "__main__":
    test_component()
