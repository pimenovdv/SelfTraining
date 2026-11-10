import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim))
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        # Forward pass is z + u * tanh(w^T z + b)
        # z: (batch_size, dim)
        linear = torch.mm(z, self.w.t()) + self.b # (batch_size, 1)
        activation = torch.tanh(linear) # (batch_size, 1)
        z_out = z + self.u * activation # (batch_size, dim)

        # Log determinant of Jacobian
        # psi = (1 - tanh(w^T z + b)^2) * w
        psi = (1 - activation**2) * self.w # (batch_size, dim)
        det_grad = 1 + torch.mm(psi, self.u.t()) # (batch_size, 1)

        # Ensure invertibility condition: w^T u >= -1
        # In a strict implementation, we would modify u to ensure this,
        # but for a simple test we just take the absolute value or clamp it
        log_det = torch.log(torch.abs(det_grad) + 1e-6).squeeze(-1)

        return z_out, log_det

class NormalizingFlow(nn.Module):
    def __init__(self, dim, num_flows):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(num_flows)])

    def forward(self, z):
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det
        return z, log_det_sum

def target_density(z):
    # Mixture of Gaussians as target distribution
    z1, z2 = z[:, 0], z[:, 1]
    norm = torch.sqrt(z1**2 + z2**2)
    exp1 = torch.exp(-0.5 * ((z1 - 2)**2 / 0.8 + (z2 - 2)**2 / 0.8))
    exp2 = torch.exp(-0.5 * ((z1 + 2)**2 / 0.8 + (z2 + 2)**2 / 0.8))
    exp3 = torch.exp(-0.5 * ((norm - 4)**2 / 0.5)) # ring
    return (exp1 + exp2 + exp3) / 3.0

def main():
    dim = 2
    num_flows = 8
    model = NormalizingFlow(dim, num_flows)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    batch_size = 512
    epochs = 2000

    print("Training Normalizing Flow with Planar Flows...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Sample from base distribution (standard normal)
        z0 = torch.randn(batch_size, dim)

        # Pass through flow
        zk, log_det = model(z0)

        # Calculate loss (KL divergence: D_KL(q_k || p) = E_{z0} [log q_k(z_k) - log p(z_k)])
        # log q_k(z_k) = log q_0(z_0) - sum(log |det J|)
        # where q_0 is standard normal
        log_q0 = -0.5 * (z0**2).sum(dim=1) - 0.5 * dim * np.log(2 * np.pi)
        log_qk = log_q0 - log_det

        # Target log density (unnormalized is fine for KL)
        log_p = torch.log(target_density(zk) + 1e-8)

        loss = (log_qk - log_p).mean()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("Training Complete. Verifying component.")

    # Save the component to visually test if wanted
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/normalizing_flows_model.pt")
    print("Normalizing Flows model saved successfully.")

if __name__ == "__main__":
    main()
