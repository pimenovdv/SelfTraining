import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InformationBottleneck(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Encoder: X -> (mu, log_var)
        self.fc_mu = nn.Linear(input_dim, hidden_dim)
        self.fc_logvar = nn.Linear(input_dim, hidden_dim)

        # Decoder / Classifier: Z -> Y
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_pred = self.decoder(z)
        return y_pred, mu, log_var

def compute_ib_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Classification loss (cross entropy for Y)
    ce_loss = nn.CrossEntropyLoss()(y_pred, y_true)

    # Information Bottleneck term: KL divergence between P(Z|X) and N(0, I)
    # This acts as an upper bound for I(Z;X)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    total_loss = ce_loss + beta * kl_divergence
    return total_loss, ce_loss, kl_divergence

def train_ib_component():
    logger.info("Initializing Information Bottleneck component training...")

    # Generate synthetic dataset
    np.random.seed(42)
    torch.manual_seed(42)

    # 2 classes, 10-dimensional input, where only first 2 dimensions matter
    n_samples = 1000
    input_dim = 10

    X = np.random.randn(n_samples, input_dim)
    # The label depends only on the first 2 features (creating a bottleneck opportunity)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    # Add noise to input
    X = X + 0.5 * np.random.randn(n_samples, input_dim)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model parameters
    hidden_dim = 2 # Z dimension
    output_dim = 2
    beta = 0.01 # Trade-off parameter

    model = InformationBottleneck(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        total_ce = 0
        total_kl = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            y_pred, mu, log_var = model(batch_x)

            loss, ce, kl = compute_ib_loss(y_pred, batch_y, mu, log_var, beta)

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_ce += ce.item()
            total_kl += kl.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_epoch_loss / len(dataloader)
            avg_ce = total_ce / len(dataloader)
            avg_kl = total_kl / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - CE: {avg_ce:.4f} - KL: {avg_kl:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred, _, _ = model(X_tensor)
        predictions = torch.argmax(y_pred, dim=1)
        accuracy = (predictions == y_tensor).float().mean().item()

    logger.info(f"Final Accuracy: {accuracy:.4f}")
    if accuracy > 0.80:
        logger.info("Information Bottleneck component verified successfully.")
    else:
        logger.warning("Accuracy lower than expected. Further tuning might be needed.")

if __name__ == "__main__":
    train_ib_component()
