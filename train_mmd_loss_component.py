import torch
import torch.nn as nn
import torch.optim as optim

class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss using an RBF kernel.
    """
    def __init__(self, gamma=1.0):
        super(MMDLoss, self).__init__()
        self.gamma = gamma

    def rbf_kernel(self, x, y):
        # x: [B1, D]
        # y: [B2, D]
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.exp(-self.gamma * dist)

    def forward(self, x, y):
        xx = self.rbf_kernel(x, x)
        yy = self.rbf_kernel(y, y)
        xy = self.rbf_kernel(x, y)
        return xx.mean() + yy.mean() - 2.0 * xy.mean()

def train_mmd_loss_component():
    """
    Trains a simple linear transformation to map source data distribution
    to a target data distribution using MMD loss.
    """
    torch.manual_seed(42)

    # Source distribution: N(0, 1)
    source_data = torch.randn(100, 2)
    # Target distribution: N(5, 1)
    target_data = torch.randn(100, 2) + 5.0

    # Model to transform source to match target
    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    mmd_loss = MMDLoss(gamma=0.5)

    initial_loss = mmd_loss(model(source_data), target_data).item()
    print(f"Initial MMD Loss: {initial_loss:.4f}")

    for epoch in range(100):
        optimizer.zero_grad()
        transformed_data = model(source_data)
        loss = mmd_loss(transformed_data, target_data)
        loss.backward()
        optimizer.step()

    final_loss = mmd_loss(model(source_data), target_data).item()
    print(f"Final MMD Loss: {final_loss:.4f}")
    return initial_loss > final_loss

if __name__ == '__main__':
    success = train_mmd_loss_component()
    if success:
        print("Successfully trained using MMD Loss!")
    else:
        print("Training failed to reduce MMD loss.")
