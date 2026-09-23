import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, output_dim=64):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.conv4 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(output_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        return x

class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder

    def forward(self, support_x, support_y, query_x):
        # support_x: [n_support, channels, height, width]
        # support_y: [n_support]
        # query_x: [n_query, channels, height, width]

        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)

        classes = torch.unique(support_y)
        n_classes = len(classes)

        # Calculate prototypes for each class
        prototypes = torch.zeros(n_classes, support_embeddings.size(-1), device=support_embeddings.device)
        for i, c in enumerate(classes):
            prototypes[i] = support_embeddings[support_y == c].mean(0)

        # Calculate distances from queries to prototypes
        dists = torch.cdist(query_embeddings, prototypes)

        # The logits are negative distances (closer is better)
        logits = -dists

        return logits

def train_few_shot_component():
    print("Initializing Few-Shot Learning (Prototypical Networks) component...")

    # 1. Setup Data (Synthetic Omniglot-like data: 1 channel, 28x28)
    # N-way K-shot learning setup
    N_way = 5
    K_shot = 5
    Q_query = 5 # queries per class

    n_support = N_way * K_shot
    n_query = N_way * Q_query

    # Synthetic support and query sets
    support_x = torch.randn(n_support, 1, 28, 28)
    support_y = torch.arange(N_way).repeat_interleave(K_shot)

    query_x = torch.randn(n_query, 1, 28, 28)
    query_y = torch.arange(N_way).repeat_interleave(Q_query)

    # 2. Setup Model
    encoder = SimpleCNN(input_channels=1, hidden_dim=64, output_dim=64)
    model = PrototypicalNetwork(encoder)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 3. Train Step
    print(f"Training Prototypical Network on {N_way}-way {K_shot}-shot task...")
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()

        # In a real scenario, we would sample different tasks (classes) each episode.
        # Here we just overfit to the synthetic task to verify the component works.
        logits = model(support_x, support_y, query_x)

        loss = criterion(logits, query_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == query_y).float().mean()
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {acc.item() * 100:.2f}%")

    print("Training complete. Component verified.")

if __name__ == "__main__":
    train_few_shot_component()
