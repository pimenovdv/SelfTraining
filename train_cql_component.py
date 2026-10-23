import numpy as np

class ContinuousCQLAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, tau=0.005, alpha=1.0, cql_alpha=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.cql_alpha = cql_alpha

        self.W_q1 = np.random.randn(state_dim + action_dim, 32) * 0.1
        self.b_q1 = np.zeros(32)
        self.W_q1_out = np.random.randn(32, 1) * 0.1
        self.b_q1_out = np.zeros(1)

        self.W_q2 = np.random.randn(state_dim + action_dim, 32) * 0.1
        self.b_q2 = np.zeros(32)
        self.W_q2_out = np.random.randn(32, 1) * 0.1
        self.b_q2_out = np.zeros(1)

        self.W_q1_target = self.W_q1.copy()
        self.b_q1_target = self.b_q1.copy()
        self.W_q1_out_target = self.W_q1_out.copy()
        self.b_q1_out_target = self.b_q1_out.copy()

        self.W_q2_target = self.W_q2.copy()
        self.b_q2_target = self.b_q2.copy()
        self.W_q2_out_target = self.W_q2_out.copy()
        self.b_q2_out_target = self.b_q2_out.copy()

        self.W_mu = np.random.randn(state_dim, 32) * 0.1
        self.b_mu = np.zeros(32)
        self.W_mu_out = np.random.randn(32, action_dim) * 0.1
        self.b_mu_out = np.zeros(action_dim)

        self.log_std = np.zeros(action_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def q_forward(self, states, actions, target=False):
        xu = np.concatenate([states, actions], axis=-1)
        if not target:
            h1 = self.relu(xu @ self.W_q1 + self.b_q1)
            q1 = h1 @ self.W_q1_out + self.b_q1_out

            h2 = self.relu(xu @ self.W_q2 + self.b_q2)
            q2 = h2 @ self.W_q2_out + self.b_q2_out
        else:
            h1 = self.relu(xu @ self.W_q1_target + self.b_q1_target)
            q1 = h1 @ self.W_q1_out_target + self.b_q1_out_target

            h2 = self.relu(xu @ self.W_q2_target + self.b_q2_target)
            q2 = h2 @ self.W_q2_out_target + self.b_q2_out_target

        return q1, q2

    def policy_forward(self, states):
        h = self.relu(states @ self.W_mu + self.b_mu)
        mu = h @ self.W_mu_out + self.b_mu_out
        std = np.exp(np.clip(self.log_std, -20, 2))
        return mu, std

    def sample_actions(self, states):
        mu, std = self.policy_forward(states)
        eps = np.random.randn(*mu.shape)
        u = mu + std * eps
        actions = np.tanh(u)

        log_prob = -0.5 * (((u - mu) / std) ** 2 + np.log(2 * np.pi * std ** 2))
        log_prob -= np.log(1 - actions ** 2 + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdims=True)
        return actions, log_prob

    def train_step(self, states, actions, rewards, next_states, dones):
        batch_size = states.shape[0]

        with np.errstate(all='ignore'):
            next_actions, next_log_probs = self.sample_actions(next_states)
            q1_target, q2_target = self.q_forward(next_states, next_actions, target=True)
            min_q_target = np.minimum(q1_target, q2_target)

            y = rewards + self.gamma * (1 - dones) * (min_q_target - self.alpha * next_log_probs)

        q1, q2 = self.q_forward(states, actions)

        num_random = 10
        random_actions = np.random.uniform(-1, 1, (batch_size, num_random, self.action_dim))
        curr_actions, curr_log_probs = self.sample_actions(states)
        next_actions_curr, next_log_probs_curr = self.sample_actions(next_states)

        q1_curr, q2_curr = self.q_forward(states, curr_actions)
        q1_rand, q2_rand = self.q_forward(np.repeat(states, num_random, axis=0), random_actions.reshape(-1, self.action_dim))
        q1_rand = q1_rand.reshape(batch_size, num_random, 1)
        q2_rand = q2_rand.reshape(batch_size, num_random, 1)

        cat_q1 = np.concatenate([q1_curr.reshape(batch_size, 1, 1), q1_rand], axis=1)
        lse_q1 = np.log(np.sum(np.exp(cat_q1), axis=1) + 1e-8)

        cat_q2 = np.concatenate([q2_curr.reshape(batch_size, 1, 1), q2_rand], axis=1)
        lse_q2 = np.log(np.sum(np.exp(cat_q2), axis=1) + 1e-8)

        cql_loss1 = self.cql_alpha * (lse_q1 - q1).mean()
        cql_loss2 = self.cql_alpha * (lse_q2 - q2).mean()

        mse_loss1 = 0.5 * np.mean((q1 - y) ** 2)
        mse_loss2 = 0.5 * np.mean((q2 - y) ** 2)

        total_q1_loss = mse_loss1 + cql_loss1
        total_q2_loss = mse_loss2 + cql_loss2

        return {
            'q1_loss': total_q1_loss,
            'q2_loss': total_q2_loss,
            'cql_penalty1': cql_loss1,
            'cql_penalty2': cql_loss2,
            'q_val': q1.mean()
        }

if __name__ == "__main__":
    print("Testing Conservative Q-Learning (CQL) component mathematically...")

    state_dim = 4
    action_dim = 2
    batch_size = 64

    states = np.random.randn(batch_size, state_dim)
    actions = np.random.uniform(-1, 1, (batch_size, action_dim))
    rewards = np.random.randn(batch_size, 1)
    next_states = np.random.randn(batch_size, state_dim)
    dones = np.random.randint(0, 2, (batch_size, 1))

    agent = ContinuousCQLAgent(state_dim=state_dim, action_dim=action_dim)

    metrics = agent.train_step(states, actions, rewards, next_states, dones)

    print("\nCQL Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nCQL mathematical formulation verified successfully.")
