import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np

class SparseCoding(nn.Module):
    def __init__(self, input_dim, dictionary_size, sparsity_penalty):
        super(SparseCoding, self).__init__()
        self.dictionary = nn.Parameter(torch.randn(input_dim, dictionary_size))
        # Normalize dictionary atoms
        with torch.no_grad():
            self.dictionary.div_(torch.norm(self.dictionary, dim=0, keepdim=True))
        self.sparsity_penalty = sparsity_penalty

    def forward(self, x, num_iters=50, lr=0.1):
        batch_size = x.size(0)
        dictionary_size = self.dictionary.size(1)

        # Initialize sparse codes
        codes = torch.zeros(batch_size, dictionary_size, requires_grad=True, device=x.device)
        optimizer = optim.SGD([codes], lr=lr)

        # Iteratively infer sparse codes using ISTA-like gradient descent
        for _ in range(num_iters):
            optimizer.zero_grad()
            reconstruction = torch.matmul(codes, self.dictionary.t())
            recon_loss = nn.functional.mse_loss(reconstruction, x)
            l1_penalty = self.sparsity_penalty * torch.mean(torch.abs(codes))
            loss = recon_loss + l1_penalty
            loss.backward()
            optimizer.step()

        return codes.detach(), torch.matmul(codes.detach(), self.dictionary.t())

def train_sparse_coding():
    print("Starting Sparse Coding training...")
    # Generate synthetic data (e.g., combination of sparse signals)
    torch.manual_seed(42)
    np.random.seed(42)

    num_samples = 1000
    input_dim = 20
    dictionary_size = 50
    sparsity_penalty = 0.1

    # Create true dictionary and sparse codes
    true_dict = torch.randn(input_dim, dictionary_size)
    true_dict.div_(torch.norm(true_dict, dim=0, keepdim=True))

    true_codes = torch.zeros(num_samples, dictionary_size)
    # 3 non-zero elements per sample
    for i in range(num_samples):
        indices = np.random.choice(dictionary_size, 3, replace=False)
        true_codes[i, indices] = torch.randn(3)

    data = torch.matmul(true_codes, true_dict.t())

    model = SparseCoding(input_dim, dictionary_size, sparsity_penalty)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 20
    batch_size = 64

    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            batch_data = data[i:i+batch_size]

            # Infer codes
            codes, recon = model(batch_data, num_iters=20, lr=0.1)

            # Update dictionary
            optimizer.zero_grad()
            reconstruction = torch.matmul(codes, model.dictionary.t())
            loss = nn.functional.mse_loss(reconstruction, batch_data)
            loss.backward()
            optimizer.step()

            # Normalize dictionary
            with torch.no_grad():
                model.dictionary.div_(torch.norm(model.dictionary, dim=0, keepdim=True))

            epoch_loss += loss.item() * batch_data.size(0)

        epoch_loss /= num_samples
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Reconstruction Loss: {epoch_loss:.4f}")

    print("Sparse Coding training completed successfully.")

if __name__ == "__main__":
    train_sparse_coding()
