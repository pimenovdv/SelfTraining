import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class LiquidStateMachine(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.1, leak_rate=0.3):
        super(LiquidStateMachine, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leak_rate = leak_rate

        # Initialize input weights
        self.W_in = nn.Linear(input_size, reservoir_size)

        # Initialize reservoir weights
        W_res = np.random.randn(reservoir_size, reservoir_size)
        # Apply sparsity
        mask = np.random.rand(reservoir_size, reservoir_size) < sparsity
        W_res = W_res * mask

        # Scale by spectral radius
        eigenvalues = np.linalg.eigvals(W_res)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            W_res = W_res * (spectral_radius / max_eigenvalue)

        self.W_res = torch.tensor(W_res, dtype=torch.float32)

        # Output weights (trainable)
        self.W_out = nn.Linear(reservoir_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize state
        state = torch.zeros(batch_size, self.reservoir_size, device=device)
        self.W_res = self.W_res.to(device)

        # Process sequence
        for t in range(seq_len):
            u = x[:, t, :]
            # Non-linear activation (tanh) for the reservoir with leak rate
            update = torch.tanh(self.W_in(u) + torch.matmul(state, self.W_res.t()))
            state = (1 - self.leak_rate) * state + self.leak_rate * update

        # Final output
        return self.W_out(state)

def main():
    print("Testing Liquid State Machine Component...")

    # Parameters
    input_size = 1
    reservoir_size = 100
    output_size = 1
    seq_len = 10
    batch_size = 32

    # Create model
    model = LiquidStateMachine(input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.2, leak_rate=1.0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Generate simple sine wave data
    x_data = torch.linspace(0, 4 * np.pi, 500).view(-1, 1)
    y_data = torch.sin(x_data)

    # Prepare sequences
    X, Y = [], []
    for i in range(len(x_data) - seq_len):
        X.append(x_data[i:i+seq_len])
        Y.append(y_data[i+seq_len])

    X = torch.stack(X)
    Y = torch.stack(Y)

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_Y in dataloader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

    print("Training complete.")
    assert total_loss/len(dataloader) < 0.2, "Loss is too high, model failed to learn."
    print("Liquid State Machine verified successfully.")

if __name__ == "__main__":
    main()
