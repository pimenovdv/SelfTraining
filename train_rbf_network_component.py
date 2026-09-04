import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RBFNetwork(nn.Module):
    def __init__(self, in_features, num_centers, out_features):
        super(RBFNetwork, self).__init__()
        self.centers = nn.Parameter(torch.Tensor(num_centers, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(num_centers))
        self.linear = nn.Linear(num_centers, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, x):
        batch_size = x.size(0)
        num_centers = self.centers.size(0)
        x_expand = x.unsqueeze(1).expand(batch_size, num_centers, -1)
        c_expand = self.centers.unsqueeze(0).expand(batch_size, num_centers, -1)
        distances = (x_expand - c_expand).pow(2).sum(2)
        sigmas = torch.exp(self.log_sigmas).unsqueeze(0).expand(batch_size, num_centers)
        phi = torch.exp(-distances / (2 * sigmas.pow(2)))
        return self.linear(phi)

if __name__ == '__main__':
    # Generate some toy data
    X = torch.randn(100, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 < 1.0).float().unsqueeze(1)

    model = RBFNetwork(in_features=2, num_centers=10, out_features=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    print(f"Final Loss: {loss.item():.4f}")
    print("RBF Network successfully trained.")
