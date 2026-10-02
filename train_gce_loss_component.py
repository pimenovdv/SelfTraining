import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class GCELoss(nn.Module):
    """
    Generalized Cross Entropy Loss.
    Combines Mean Absolute Error (MAE) and Categorical Cross Entropy (CCE)
    for robustness against noisy labels.
    """
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        # targets should be class indices
        probs = F.softmax(logits, dim=-1)
        # Gather probabilities for true labels
        true_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        # Calculate GCE
        loss = (1.0 - torch.pow(true_probs, self.q)) / self.q
        return loss.mean()

def train_gce_loss_component():
    torch.manual_seed(42)
    q_value = 0.7
    criterion = GCELoss(q=q_value)

    # Simulate a classification task with noisy labels
    # 3 classes, 10 samples
    logits = torch.randn(10, 3, requires_grad=True)
    targets = torch.randint(0, 3, (10,))

    optimizer = torch.optim.SGD([logits], lr=0.1)

    losses = []

    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    results = {
        "component": "Generalized Cross Entropy Loss (GCELoss)",
        "q_value": q_value,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_trajectory": losses
    }

    os.makedirs("results", exist_ok=True)
    with open("results/gce_loss_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Initial Loss: {losses[0]:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")
    print("Optimization successful.")

if __name__ == "__main__":
    train_gce_loss_component()
