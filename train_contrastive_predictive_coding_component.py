import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        z = self.fc2(x)
        return z

class Autoregressive(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super(Autoregressive, self).__init__()
        self.gru = nn.GRU(hidden_dim, context_dim, batch_first=True)

    def forward(self, z):
        c, _ = self.gru(z)
        return c

class CPCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim, k_steps):
        super(CPCModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.ar = Autoregressive(hidden_dim, context_dim)
        self.k_steps = k_steps
        self.predictors = nn.ModuleList([
            nn.Linear(context_dim, hidden_dim) for _ in range(k_steps)
        ])

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        # Flatten for encoder
        x_flat = x.contiguous().view(-1, x.size(-1))
        z_flat = self.encoder(x_flat)
        z = z_flat.view(batch_size, seq_len, -1)

        c = self.ar(z)
        return z, c

def info_nce_loss(z, c, predictors, k_steps):
    batch_size, seq_len, hidden_dim = z.size()
    total_loss = 0.0

    for k in range(1, k_steps + 1):
        if seq_len - k <= 0:
            break

        # Context at time t
        c_t = c[:, :-k, :]
        # Target at time t+k
        z_tk = z[:, k:, :]

        # Predicted target
        pred_z_tk = predictors[k-1](c_t)

        # Reshape to calculate dot products
        pred_z_tk = pred_z_tk.contiguous().view(-1, hidden_dim)
        z_tk = z_tk.contiguous().view(-1, hidden_dim)

        # Calculate scores
        # Positive pairs: diagonal of dot product matrix
        # Negative pairs: off-diagonal elements
        scores = torch.matmul(pred_z_tk, z_tk.t())

        # Labels are the indices 0, 1, 2, ...
        labels = torch.arange(scores.size(0)).to(scores.device)

        loss = F.cross_entropy(scores, labels)
        total_loss += loss

    return total_loss / k_steps

def test_cpc():
    torch.manual_seed(42)
    batch_size = 32
    seq_len = 20
    input_dim = 10
    hidden_dim = 16
    context_dim = 32
    k_steps = 3

    model = CPCModel(input_dim, hidden_dim, context_dim, k_steps)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training Contrastive Predictive Coding (CPC) Component...")

    # Dummy data
    x = torch.randn(batch_size, seq_len, input_dim)

    for epoch in range(10):
        optimizer.zero_grad()
        z, c = model(x)
        loss = info_nce_loss(z, c, model.predictors, k_steps)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("CPC component trained successfully.")

if __name__ == "__main__":
    test_cpc()
