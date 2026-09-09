import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class A3CNet(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(A3CNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, num_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

def run_experiment():
    print("Initializing A3C component...")
    input_dim = 4
    num_actions = 2
    model = A3CNet(input_dim, num_actions)

    # Mock data for testing
    state = torch.randn(1, input_dim)
    policy_logits, value = model(state)

    print("State:", state.shape)
    print("Policy Logits:", policy_logits.shape)
    print("Value:", value.shape)

    assert policy_logits.shape == (1, num_actions), "Policy logits shape mismatch"
    assert value.shape == (1, 1), "Value shape mismatch"
    print("A3C component test passed!")

if __name__ == "__main__":
    run_experiment()
