import torch
import torch.nn as nn
import torch.optim as optim
import math

class SimpleDiffusionModel(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim), # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        t = t.unsqueeze(-1)
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

def train_diffusion():
    torch.manual_seed(42)
    # Generate some simple 2D data (e.g. circle)
    theta = torch.rand(1000) * 2 * math.pi
    data = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    model = SimpleDiffusionModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100
    batch_size = 128

    for epoch in range(num_epochs):
        perm = torch.randperm(len(data))
        for i in range(0, len(data), batch_size):
            batch = data[perm[i:i+batch_size]]
            t = torch.rand(len(batch))
            noise = torch.randn_like(batch)

            # Continuous time formulation [0,1]
            alpha_bar = 1 - t.unsqueeze(-1)
            noisy_data = torch.sqrt(alpha_bar) * batch + torch.sqrt(1 - alpha_bar) * noise

            # Predict noise
            pred_noise = model(noisy_data, t)

            loss = nn.MSELoss()(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Diffusion Model training complete. Final Loss:", loss.item())

if __name__ == "__main__":
    train_diffusion()
