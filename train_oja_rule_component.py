"""
Oja's Rule (Hebbian Learning) Component

This script explores unsupervised representation learning using Oja's rule,
a variant of Hebbian learning that introduces a normalization term to prevent
weights from growing indefinitely. It inherently performs Principal Component
Analysis (PCA) by finding the leading principal components of the input data.

Mathematical Foundation:
1. Standard Hebbian Learning: ΔW = η * y * x^T
2. Oja's Rule: ΔW = η * (y * x^T - y^2 * W)
   where η is the learning rate, y is the output, x is the input, and W is the weight matrix.

The subtractive term (- y^2 * W) acts as a weight decay that is proportional
to the output variance, naturally constraining the weight vectors to unit norm
and causing them to converge to the principal eigenvectors of the input
covariance matrix.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np

class OjasRuleLinear(nn.Module):
    """
    A linear layer that learns its weights using Oja's rule (unsupervised).
    """
    def __init__(self, in_features, out_features, lr=0.01):
        super().__init__()
        # Initialize weights randomly, but small
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.lr = lr

    def forward(self, x):
        """
        Forward pass and weight update via Oja's rule.
        x: (batch_size, in_features)
        """
        # Linear projection: y = Wx
        y = torch.matmul(x, self.weight.t())

        # Apply Oja's rule for weight update if in training mode
        if self.training:
            with torch.no_grad():
                # Outer product of output and input: y * x^T
                # yx: (batch_size, out_features, in_features)
                yx = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))

                # Normalization term: y^2 * W
                # y2: (batch_size, out_features, 1)
                # w: (1, out_features, in_features)
                # y2w: (batch_size, out_features, in_features)
                y2 = (y**2).unsqueeze(2)
                w = self.weight.unsqueeze(0)
                y2w = y2 * w

                # Average over batch: ΔW = mean(y*x^T - y^2*W)
                dw = torch.mean(yx - y2w, dim=0)

                # Update weights
                self.weight.add_(self.lr * dw)

        return y


def run_experiment():
    print("Starting Oja's Rule (Hebbian Learning) Component Training...")

    # Generate synthetic dataset (e.g., points in 3D that mostly lie on a 2D plane)
    np.random.seed(42)
    torch.manual_seed(42)

    num_samples = 2000
    in_dim = 3
    out_dim = 2 # Extract top 2 principal components

    # Create covariance matrix with distinct eigenvalues
    cov = np.array([[5.0, 2.0, 0.5],
                    [2.0, 2.0, 0.2],
                    [0.5, 0.2, 0.5]])

    # Generate data
    X_np = np.random.multivariate_normal([0, 0, 0], cov, num_samples)

    # Calculate true PCA for comparison
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    true_components = eigenvectors[:, idx][:, :out_dim].T

    X_tensor = torch.FloatTensor(X_np)

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = OjasRuleLinear(in_dim, out_dim, lr=0.01)

    print(f"Dataset generated. Input dimension: {in_dim}, Output dimension (PCs): {out_dim}")
    print(f"True Top {out_dim} Principal Components:\n{true_components}")

    # Training Loop
    epochs = 20
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            x_batch = batch[0]
            # Forward pass updates the weights internally
            _ = model(x_batch)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] completed.")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Evaluate learned weights
    learned_weights = model.weight.detach().numpy()

    # Normalize learned weights to unit length for comparison
    norms = np.linalg.norm(learned_weights, axis=1, keepdims=True)
    learned_weights_normalized = learned_weights / (norms + 1e-8)

    print(f"\nLearned Weights (normalized):\n{learned_weights_normalized}")

    # Check orthogonality (W * W^T should be close to identity)
    ortho_check = np.dot(learned_weights_normalized, learned_weights_normalized.T)
    print(f"\nOrthogonality check (W * W^T):\n{ortho_check}")

    # The learned weights should span the same subspace as the true top PCs
    # (they might be rotated or sign-flipped versions of the true PCs)

    # Save a small artifact
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/oja_rule_model.pt")
    print("Model saved to results/oja_rule_model.pt")

if __name__ == "__main__":
    run_experiment()
