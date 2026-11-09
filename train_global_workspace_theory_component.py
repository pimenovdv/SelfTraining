import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpecialistModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.confidence_scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        confidence = self.confidence_scorer(x)
        return features, confidence

class GlobalWorkspaceModel(nn.Module):
    def __init__(self, num_modules, input_dim, hidden_dim, workspace_dim, output_dim):
        super().__init__()
        self.modules_list = nn.ModuleList([
            SpecialistModule(input_dim, hidden_dim, workspace_dim) for _ in range(num_modules)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(workspace_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, num_modules, input_dim]
        # In a more general case, each module could receive different inputs.
        # Here we assume a split input or identical inputs. We'll pass the same x to all modules for simplicity,
        # or slice it if we want independent inputs.

        batch_size = x.size(0)

        features_list = []
        confidences_list = []

        for i, mod in enumerate(self.modules_list):
            feat, conf = mod(x[:, i, :])
            features_list.append(feat.unsqueeze(1)) # [batch, 1, workspace_dim]
            confidences_list.append(conf) # [batch, 1]

        all_features = torch.cat(features_list, dim=1) # [batch, num_modules, workspace_dim]
        all_confidences = torch.cat(confidences_list, dim=1) # [batch, num_modules]

        # Workspace selection via softmax (competition for the global workspace)
        attention_weights = torch.softmax(all_confidences, dim=-1) # [batch, num_modules]

        # Broadcast: compute the expected workspace representation
        workspace_representation = torch.bmm(attention_weights.unsqueeze(1), all_features).squeeze(1) # [batch, workspace_dim]

        output = self.decoder(workspace_representation)
        return output

def generate_synthetic_data(num_samples=1000, num_modules=4, input_dim=10, output_dim=1):
    X = torch.randn(num_samples, num_modules, input_dim)

    # Target depends on a specific module based on some rule
    # e.g., if sum of first features is positive, rely on module 0, else module 1
    y = torch.zeros(num_samples, output_dim)
    for i in range(num_samples):
        if X[i, 0, 0].item() > 0:
            y[i] = X[i, 0, :].sum() * 0.5
        else:
            y[i] = X[i, 1, :].sum() * -0.5

    return X, y

def main():
    logger.info("Starting Global Workspace Theory component training...")

    num_modules = 4
    input_dim = 10
    hidden_dim = 32
    workspace_dim = 16
    output_dim = 1
    batch_size = 32
    epochs = 20
    learning_rate = 0.001

    X, y = generate_synthetic_data(num_samples=2000, num_modules=num_modules, input_dim=input_dim, output_dim=output_dim)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GlobalWorkspaceModel(num_modules, input_dim, hidden_dim, workspace_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds.")
    logger.info("Successfully completed Global Workspace Theory training.")

if __name__ == "__main__":
    main()
