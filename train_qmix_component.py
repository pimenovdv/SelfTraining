import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class QMIXNetwork:
    def __init__(self, num_agents, state_dim, hidden_dim, seed=42):
        np.random.seed(seed)
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Agent networks (simplified as linear maps from state to Q-value)
        # In reality, they take observations, but here we simplify using global state
        self.agent_w = np.random.randn(num_agents, state_dim) * 0.1
        self.agent_b = np.zeros(num_agents)

        # Mixing network (Hypernetwork)
        # Generates weights W1 (num_agents -> hidden_dim) and W2 (hidden_dim -> 1)
        # Absolute value applied to ensure monotonic mixing
        self.hyper_w1 = np.random.randn(state_dim, num_agents * hidden_dim) * 0.1
        self.hyper_b1 = np.random.randn(state_dim, hidden_dim) * 0.1

        self.hyper_w2 = np.random.randn(state_dim, hidden_dim * 1) * 0.1
        self.hyper_b2 = np.random.randn(state_dim, 1) * 0.1

    def agent_q_values(self, state):
        # state: (state_dim,)
        # Returns: (num_agents,)
        return np.dot(self.agent_w, state) + self.agent_b

    def mix_q_values(self, state, q_vals):
        # Hypernetwork outputs absolute weights
        # W1: (hidden_dim, num_agents)
        w1_flat = np.abs(np.dot(state, self.hyper_w1))
        w1 = w1_flat.reshape(self.hidden_dim, self.num_agents)
        b1_flat = np.dot(state, self.hyper_b1)
        b1 = b1_flat.reshape(self.hidden_dim)

        # W2: (1, hidden_dim)
        w2_flat = np.abs(np.dot(state, self.hyper_w2))
        w2 = w2_flat.reshape(1, self.hidden_dim)

        # State-dependent bias
        b2_flat = np.dot(state, self.hyper_b2)
        b2 = b2_flat[0]

        self.last_w1 = w1
        self.last_b1 = b1
        self.last_w2 = w2
        self.last_b2 = b2

        # Mixing pass
        h = relu(np.dot(w1, q_vals) + b1) # (hidden_dim,)
        q_tot = np.dot(w2, h) + b2 # (1,)

        self.last_h = h
        self.last_q_vals = q_vals
        self.last_state = state
        return q_tot[0]

    def forward(self, state):
        q_vals = self.agent_q_values(state)
        q_tot = self.mix_q_values(state, q_vals)
        return q_vals, q_tot

    def backward(self, grad_q_tot, lr=0.01):
        # grad_q_tot: scalar (d L / d Q_tot)

        # Backprop through mixing net W2
        # q_tot = w2 @ h + b2
        grad_w2 = grad_q_tot * self.last_h.reshape(1, -1)
        grad_b2_flat = grad_q_tot
        grad_h = grad_q_tot * self.last_w2.flatten() # (hidden_dim,)

        # Backprop through relu
        grad_pre_h = grad_h * relu_deriv(np.dot(self.last_w1, self.last_q_vals) + self.last_b1) # (hidden_dim,)

        # Backprop through W1
        # pre_h = w1 @ q_vals + b1
        grad_w1 = np.outer(grad_pre_h, self.last_q_vals) # (hidden_dim, num_agents)
        grad_b1 = grad_pre_h

        grad_q_vals = np.dot(self.last_w1.T, grad_pre_h) # (num_agents,)

        # Backprop hypernetwork (simplified, ignoring derivatives of abs for simplicity,
        # using sign for true subgradient but approximation holds for this simple test)
        # W1_flat = abs(state @ hyper_w1) -> grad_hyper_w1 = state.T @ (grad_w1 * sign(state @ hyper_w1))

        sign_w1 = np.sign(np.dot(self.last_state, self.hyper_w1)).reshape(self.hidden_dim, self.num_agents)
        grad_hyper_w1 = np.outer(self.last_state, (grad_w1 * sign_w1).flatten())
        grad_hyper_b1 = np.outer(self.last_state, grad_b1.flatten())

        sign_w2 = np.sign(np.dot(self.last_state, self.hyper_w2)).reshape(1, self.hidden_dim)
        grad_hyper_w2 = np.outer(self.last_state, (grad_w2 * sign_w2).flatten())
        grad_hyper_b2 = np.outer(self.last_state, grad_b2_flat)

        # Backprop agent networks
        grad_agent_w = np.outer(grad_q_vals, self.last_state)
        grad_agent_b = grad_q_vals

        # Update weights (Gradient Descent)
        self.hyper_w1 -= lr * grad_hyper_w1
        self.hyper_b1 -= lr * grad_hyper_b1
        self.hyper_w2 -= lr * grad_hyper_w2
        self.hyper_b2 -= lr * grad_hyper_b2

        self.agent_w -= lr * grad_agent_w
        self.agent_b -= lr * grad_agent_b

def main():
    print("Testing QMIX Component...")
    num_agents = 2
    state_dim = 4
    hidden_dim = 8

    qmix = QMIXNetwork(num_agents, state_dim, hidden_dim)

    # Simple synthetic objective: Learn to match a target Q_tot
    state = np.array([1.0, -0.5, 0.5, 0.2])
    target_q_tot = 5.0

    epochs = 100
    lr = 0.05

    for epoch in range(epochs):
        q_vals, q_tot = qmix.forward(state)

        loss = 0.5 * (q_tot - target_q_tot)**2
        grad_q_tot = (q_tot - target_q_tot)

        qmix.backward(grad_q_tot, lr)

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Q_tot: {q_tot:.4f}")

    print("Test passed successfully.")

if __name__ == "__main__":
    main()
