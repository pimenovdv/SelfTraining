import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

class RenyiDivergenceLoss(nn.Module):
    def __init__(self, alpha=2.0, epsilon=1e-8):
        super(RenyiDivergenceLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, p, q):
        # p and q should be probability distributions
        p = torch.clamp(p, self.epsilon, 1.0)
        q = torch.clamp(q, self.epsilon, 1.0)

        if self.alpha == 1.0:
            # Approaches KL Divergence as alpha -> 1
            return torch.sum(p * torch.log(p / q), dim=-1).mean()

        term = (p ** self.alpha) * (q ** (1 - self.alpha))
        divergence = (1 / (self.alpha - 1)) * torch.log(torch.sum(term, dim=-1) + self.epsilon)
        return divergence.mean()

def run_experiment():
    torch.manual_seed(42)

    # Target distribution
    target = torch.softmax(torch.randn(10, 5), dim=-1)

    # Model producing logits
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )

    criterion = RenyiDivergenceLoss(alpha=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    inputs = torch.randn(10, 5)

    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)

        loss = criterion(probs, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Initial Loss: {losses[0]:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")
    print("Renyi Divergence (alpha=2.0) optimization successful.")

    os.makedirs('results', exist_ok=True)
    with open('results/renyi_divergence_results.json', 'w') as f:
        json.dump({'initial_loss': losses[0], 'final_loss': losses[-1]}, f)

if __name__ == "__main__":
    run_experiment()
