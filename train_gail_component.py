import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))

class Discriminator:
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.W1 = np.random.randn(state_dim + action_dim, 16) * 0.1
        self.b1 = np.zeros((1, 16))
        self.W2 = np.random.randn(16, 1) * 0.1
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def forward(self, state, action):
        x = np.concatenate([state, action], axis=-1)
        h = np.maximum(0, x.dot(self.W1) + self.b1)
        z = h.dot(self.W2) + self.b2
        return sigmoid(z), h

    def train_step(self, expert_state, expert_action, policy_state, policy_action):
        expert_out, expert_h = self.forward(expert_state, expert_action)
        policy_out, policy_h = self.forward(policy_state, policy_action)

        expert_loss = -np.mean(np.log(expert_out + 1e-8))
        policy_loss = -np.mean(np.log(1.0 - policy_out + 1e-8))

        d_expert_z = (expert_out - 1.0) / len(expert_state)
        d_policy_z = policy_out / len(policy_state)

        dW2_exp = expert_h.T.dot(d_expert_z)
        db2_exp = np.sum(d_expert_z, axis=0, keepdims=True)
        d_h_exp = d_expert_z.dot(self.W2.T)
        d_h_exp[expert_h <= 0] = 0
        expert_x = np.concatenate([expert_state, expert_action], axis=-1)
        dW1_exp = expert_x.T.dot(d_h_exp)
        db1_exp = np.sum(d_h_exp, axis=0, keepdims=True)

        dW2_pol = policy_h.T.dot(d_policy_z)
        db2_pol = np.sum(d_policy_z, axis=0, keepdims=True)
        d_h_pol = d_policy_z.dot(self.W2.T)
        d_h_pol[policy_h <= 0] = 0
        policy_x = np.concatenate([policy_state, policy_action], axis=-1)
        dW1_pol = policy_x.T.dot(d_h_pol)
        db1_pol = np.sum(d_h_pol, axis=0, keepdims=True)

        self.W1 -= self.lr * (dW1_exp + dW1_pol)
        self.b1 -= self.lr * (db1_exp + db1_pol)
        self.W2 -= self.lr * (dW2_exp + dW2_pol)
        self.b2 -= self.lr * (db2_exp + db2_pol)

        return expert_loss + policy_loss

class Policy:
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.W = np.random.randn(state_dim, action_dim) * 0.1
        self.b = np.zeros((1, action_dim))
        self.lr = lr

    def forward(self, state):
        return state.dot(self.W) + self.b

    def train_step_differentiable(self, state, discriminator):
        action = self.forward(state)
        x = np.concatenate([state, action], axis=-1)

        h = np.maximum(0, x.dot(discriminator.W1) + discriminator.b1)
        z = h.dot(discriminator.W2) + discriminator.b2
        D_out = sigmoid(z)

        d_z = (D_out - 1.0) / len(state)

        d_h = d_z.dot(discriminator.W2.T)
        d_h[h <= 0] = 0

        d_x = d_h.dot(discriminator.W1.T)

        d_action = d_x[:, state.shape[1]:]

        dW = state.T.dot(d_action)
        db = np.sum(d_action, axis=0, keepdims=True)

        self.W -= self.lr * dW
        self.b -= self.lr * db

def test_gail():
    np.random.seed(42)
    state_dim = 2
    action_dim = 1

    expert_W = np.array([[1.0], [0.5]])

    def get_expert_data(num_samples):
        states = np.random.randn(num_samples, state_dim)
        actions = states.dot(expert_W) + np.random.randn(num_samples, action_dim) * 0.1
        return states, actions

    discriminator = Discriminator(state_dim, action_dim, lr=0.05)
    policy = Policy(state_dim, action_dim, lr=0.05)

    epochs = 4000
    batch_size = 256

    print("Training Generative Adversarial Imitation Learning (GAIL)...")
    for epoch in range(epochs):
        expert_s, expert_a = get_expert_data(batch_size)
        policy_s = np.random.randn(batch_size, state_dim)

        policy_a = policy.forward(policy_s)
        policy_a_noisy = policy_a + np.random.randn(batch_size, action_dim) * 0.1

        for _ in range(2):
            d_loss = discriminator.train_step(expert_s, expert_a, policy_s, policy_a_noisy)

        for _ in range(1):
            policy.train_step_differentiable(policy_s, discriminator)

        if epoch % 500 == 0:
            w_err = np.linalg.norm(policy.W - expert_W)
            print(f"Epoch {epoch}: D Loss = {d_loss:.4f}, Policy W Error = {w_err:.4f}")

    final_w_err = np.linalg.norm(policy.W - expert_W)
    print(f"Final Policy W Error: {final_w_err:.4f}")
    assert final_w_err < 0.2, f"Policy did not converge to expert weights. Error: {final_w_err}"
    print("GAIL successfully trained policy to imitate expert!")

if __name__ == "__main__":
    test_gail()
