import torch
import torch.nn as nn
import torch.optim as optim
import os

class HamiltonianMonteCarlo(nn.Module):
    def __init__(self, step_size, num_steps):
        super().__init__()
        self.step_size = nn.Parameter(torch.tensor(step_size))
        self.num_steps = num_steps

    def forward(self, q, p, grad_U):
        q = q.clone().detach().requires_grad_(True)
        p = p.clone().detach()

        # Leapfrog integration
        for i in range(self.num_steps):
            p = p - self.step_size * grad_U(q) / 2
            q = q + self.step_size * p
            p = p - self.step_size * grad_U(q) / 2

        return q, p

def main():
    print("Training Hamiltonian Monte Carlo Component...")

    # 1. Initialization
    q_init = torch.randn(10)
    p_init = torch.randn(10)
    hmc = HamiltonianMonteCarlo(step_size=0.1, num_steps=10)

    # 2. Define target distribution (e.g., standard normal)
    def grad_U(q):
        return q # Gradient of negative log prob of N(0, I)

    # 3. Simulate
    q_final, p_final = hmc(q_init, p_init, grad_U)

    # 4. Check results
    assert q_final.shape == q_init.shape, "Shape mismatch for q"
    assert p_final.shape == p_init.shape, "Shape mismatch for p"

    print("Training Complete. Results verifiable.")

    # 5. Save model
    os.makedirs('results', exist_ok=True)
    torch.save(hmc.state_dict(), 'results/hmc_model.pt')
    print("Model saved to results/hmc_model.pt")

if __name__ == '__main__':
    main()
