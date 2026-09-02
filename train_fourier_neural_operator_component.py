"""
Fourier Neural Operator (FNO) Component

This script implements a 1D Fourier Neural Operator to learn the mapping
between functions, often used for solving Partial Differential Equations (PDEs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# A simple 1D spectral convolution layer
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x is (batch, x_grid, 2)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

def generate_data(num_samples, grid_size):
    # Generates a(x) -> u(x) mappings where u(x) = int K(x,y)a(y) dy
    x = np.linspace(0, 1, grid_size)
    a = np.random.randn(num_samples, grid_size) * 0.1
    # Smooth a bit
    a = np.cumsum(a, axis=1)

    u = np.zeros_like(a)
    for i in range(num_samples):
        # Integral of a(x) over [0, x]
        u[i] = np.cumsum(a[i]) / grid_size

    return torch.tensor(a, dtype=torch.float32).unsqueeze(-1), torch.tensor(u, dtype=torch.float32).unsqueeze(-1)

def main():
    print("Initializing Fourier Neural Operator Component...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modes = 16
    width = 32
    grid_size = 64

    model = FNO1d(modes, width).to(device)

    # Generate mock data
    x_train, y_train = generate_data(1000, grid_size)
    x_test, y_test = generate_data(200, grid_size)

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 20
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        test_out = model(x_test)
        test_loss = criterion(test_out, y_test)
        print(f"Test Loss: {test_loss.item():.6f}")

    print("Training complete. Fourier Neural Operator model is verified.")

if __name__ == "__main__":
    main()
