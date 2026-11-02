import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Environment:
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 2

    def reset(self):
        return np.zeros(self.state_dim)

    def step(self, action):
        # Target action is [1, 1]
        reward = -np.sum((action - 1) ** 2)
        next_state = np.zeros(self.state_dim)
        done = True
        return next_state, reward, done

class NACPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mean = self.fc(x)
        std = torch.exp(self.log_std)
        return mean, std

def get_fisher_vector_product_nac(policy, states, p, cg_damping=0.1):
    p.detach()
    mean, std = policy(states)
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

def conjugate_gradient_nac(policy, states, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        z = get_fisher_vector_product_nac(policy, states, p)
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

def train_nac():
    env = Environment()
    policy = NACPolicyNetwork(env.state_dim, env.action_dim)
    value_net = nn.Linear(env.state_dim, 1)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.1)

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

        # Critic update (Value function)
        values = value_net(states_tensor).squeeze(-1)
        advantages = rewards_tensor - values
        value_loss = (advantages ** 2).mean()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Actor update (Natural Gradient using advantages)
        mean, std = policy(states_tensor)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions_tensor).sum(axis=-1)

        actor_loss = -(log_probs * advantages.detach()).mean()

        grads = torch.autograd.grad(actor_loss, policy.parameters())
        flat_grads = torch.cat([g.view(-1) for g in grads]).detach()

        step_dir = conjugate_gradient_nac(policy, states_tensor, flat_grads)

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
    train_nac()
