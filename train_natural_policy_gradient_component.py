import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 2

    def reset(self):
        return np.zeros(self.state_dim)

    def step(self, action):
        # A simple environment where the goal is to output an action of [1, 1]
        reward = -np.sum((action - 1) ** 2)
        next_state = np.zeros(self.state_dim) # State is constant for simplicity in this NPG demo
        done = False # Always return done = True for this single-step bandit-like setup
        return next_state, reward, True

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, action_dim)
        # Log std is a learnable parameter independent of state
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mean = self.fc(x)
        std = torch.exp(self.log_std)
        return mean, std

def get_fisher_vector_product(policy, states, actions, p, cg_damping=0.1):
    p.detach()
    # Compute kl divergence between current policy and itself (which is 0)
    # But we need the gradients
    mean, std = policy(states)

    # KL(N(m, s) || N(m_old, s_old))
    # Since they are the same, kl is 0, but second derivative is Fisher info
    old_mean, old_std = mean.detach(), std.detach()

    kl = torch.sum(
        torch.log(old_std / std) + (std**2 + (mean - old_mean)**2) / (2.0 * old_std**2) - 0.5,
        dim=-1
    ).mean()

    kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_kl_grad = torch.cat([g.view(-1) for g in kl_grad])

    kl_grad_p = (flat_kl_grad * p).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, policy.parameters())
    flat_kl_hessian_p = torch.cat([g.view(-1) for g in kl_hessian_p]).detach()

    return flat_kl_hessian_p + cg_damping * p

def conjugate_gradient(policy, states, actions, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        z = get_fisher_vector_product(policy, states, actions, p)
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

def train_npg():
    env = Environment()
    policy = PolicyNetwork(env.state_dim, env.action_dim)

    epochs = 100
    batch_size = 100
    learning_rate = 0.1

    for epoch in range(epochs):
        states_list = []
        actions_list = []
        rewards_list = []

        for _ in range(batch_size):
            state = env.reset()
            state_tensor = torch.FloatTensor(state)

            mean, std = policy(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

            next_state, reward, done = env.step(action.numpy())

            states_list.append(state)
            actions_list.append(action.numpy())
            rewards_list.append(reward)

        states_tensor = torch.FloatTensor(np.array(states_list))
        actions_tensor = torch.FloatTensor(np.array(actions_list))
        rewards_tensor = torch.FloatTensor(np.array(rewards_list))

        # Standardize rewards for baseline
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        mean, std = policy(states_tensor)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions_tensor).sum(axis=-1)

        loss = -(log_probs * rewards_tensor).mean()

        grads = torch.autograd.grad(loss, policy.parameters())
        flat_grads = torch.cat([g.view(-1) for g in grads]).detach()

        step_dir = conjugate_gradient(policy, states_tensor, actions_tensor, flat_grads)

        # NPG update
        idx = 0
        for param in policy.parameters():
            numel = param.numel()
            param.data -= learning_rate * step_dir[idx:idx+numel].view(param.shape)
            idx += numel

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Mean Reward = {np.mean(rewards_list):.4f}")

    print("\nEvaluating final policy:")
    state = env.reset()
    state_tensor = torch.FloatTensor(state)
    mean, std = policy(state_tensor)
    print(f"State: {state} -> Action Mean: {mean.detach().numpy()}")

if __name__ == "__main__":
    train_npg()
