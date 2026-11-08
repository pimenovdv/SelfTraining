import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ModelPredictiveControl(nn.Module):
    def __init__(self, state_dim, action_dim, horizon):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        # Simple linear dynamics: x_{t+1} = A x_t + B u_t
        self.A = nn.Parameter(torch.eye(state_dim) + 0.1 * torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(0.1 * torch.randn(state_dim, action_dim))

    def forward(self, state, action_sequence):
        # state: (batch_size, state_dim)
        # action_sequence: (batch_size, horizon, action_dim)
        batch_size = state.shape[0]
        states = [state]
        current_state = state
        for t in range(self.horizon):
            u = action_sequence[:, t, :]
            next_state = current_state @ self.A.T + u @ self.B.T
            states.append(next_state)
            current_state = next_state
        return torch.stack(states, dim=1) # (batch_size, horizon+1, state_dim)

def test_mpc():
    batch_size = 32
    state_dim = 4
    action_dim = 2
    horizon = 10

    mpc = ModelPredictiveControl(state_dim, action_dim, horizon)
    initial_state = torch.randn(batch_size, state_dim)

    # Target state is the origin
    target_state = torch.zeros(batch_size, state_dim)

    # Action sequence as optimizable parameters
    action_sequence = torch.zeros(batch_size, horizon, action_dim, requires_grad=True)

    optimizer = optim.Adam([action_sequence], lr=0.1)

    epochs = 150
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Predict trajectory
        trajectory = mpc(initial_state, action_sequence)

        # Cost function: distance to target state + action penalty
        state_cost = torch.mean(torch.norm(trajectory[:, -1, :] - target_state, dim=1))
        action_cost = 0.01 * torch.mean(torch.norm(action_sequence, dim=2))
        loss = state_cost + action_cost

        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    assert loss.item() < 1.5, f"MPC failed to converge, final loss: {loss.item()}"
    print("Model Predictive Control Component test passed.")

if __name__ == "__main__":
    test_mpc()
