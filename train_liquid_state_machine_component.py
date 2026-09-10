import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import time

class LiquidStateMachine(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.1, leak_rate=0.1):
        super(LiquidStateMachine, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leak_rate = leak_rate

        # Input weights
        self.W_in = nn.Parameter(torch.randn(reservoir_size, input_size) * 0.1, requires_grad=False)

        # Reservoir weights (fixed, random sparse matrix)
        W_res = torch.randn(reservoir_size, reservoir_size)
        W_res[torch.rand(reservoir_size, reservoir_size) > sparsity] = 0

        # Scale by spectral radius
        eigenvalues = torch.linalg.eigvals(W_res)
        max_eigenvalue = torch.max(torch.abs(eigenvalues)).item()
        if max_eigenvalue > 0:
            W_res = W_res * (spectral_radius / max_eigenvalue)

        self.W_res = nn.Parameter(W_res, requires_grad=False)

        # Readout weights (trained)
        self.W_out = nn.Linear(reservoir_size, output_size)

    def forward(self, x, h_prev=None):
        batch_size, seq_len, _ = x.size()

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.reservoir_size, device=x.device)

        outputs = []
        h_t = h_prev

        for t in range(seq_len):
            x_t = x[:, t, :]
            # Update state with leak rate (Euler method for continuous time approximation)
            h_update = torch.tanh(torch.matmul(x_t, self.W_in.t()) + torch.matmul(h_t, self.W_res.t()))
            h_t = (1 - self.leak_rate) * h_t + self.leak_rate * h_update
            outputs.append(h_t.unsqueeze(1))

        # Concatenate outputs over time
        all_states = torch.cat(outputs, dim=1)

        # Readout from the final state or all states depending on task
        # Here we do a simple readout from all states
        out = self.W_out(all_states)

        return out, h_t

def train_and_evaluate():
    # Simple temporal task: sequence prediction (sine wave)
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate data: a sine wave
    t = np.linspace(0, 100, 1000)
    data = np.sin(t)

    # Prepare sequences
    seq_len = 20
    X = []
    Y = []
    for i in range(len(data) - seq_len - 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+1:i+seq_len+1])

    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)

    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Model
    model = LiquidStateMachine(input_size=1, reservoir_size=50, output_size=1, leak_rate=0.3)

    # In Echo State Networks (closely related to LSM), the readout is usually trained via linear regression
    # But here we'll use gradient descent for simplicity of implementation in PyTorch
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.W_out.parameters(), lr=0.01)

    epochs = 100
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out, _ = model(X_train)
        loss = criterion(out, Y_train)
        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_out, _ = model(X_test)
        test_loss = criterion(test_out, Y_test).item()

    print(f"Liquid State Machine Training Complete. Test Loss: {test_loss:.4f}")

    results = {
        "model": "Liquid State Machine (LSM) Component",
        "test_loss": float(test_loss),
        "training_time": float(training_time),
        "status": "success",
        "description": "Implemented a Liquid State Machine, a type of reservoir computing network, for a continuous sequence prediction task."
    }

    os.makedirs("results", exist_ok=True)
    with open("results/lsm_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    train_and_evaluate()
