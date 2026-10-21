import numpy as np

class C51:
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=-10, v_max=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.z = np.linspace(v_min, v_max, num_atoms)

        # Simple linear network for state -> logits of distributions
        np.random.seed(42)
        self.W = np.random.randn(state_dim, action_dim * num_atoms) * 0.1
        self.b = np.zeros(action_dim * num_atoms)

        # Target network
        self.target_W = np.copy(self.W)
        self.target_b = np.copy(self.b)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def get_action(self, state):
        state = state.reshape(1, -1)
        logits = np.dot(state, self.W) + self.b
        logits = logits.reshape(1, self.action_dim, self.num_atoms)
        probs = self._softmax(logits)

        q_values = np.sum(probs * self.z, axis=2)
        return np.argmax(q_values[0])

    def train(self, states, actions, rewards, next_states, dones, gamma=0.99, lr=0.01):
        batch_size = len(states)

        # Compute next distributions with target network
        next_logits = np.dot(next_states, self.target_W) + self.target_b
        next_logits = next_logits.reshape(batch_size, self.action_dim, self.num_atoms)
        next_probs = self._softmax(next_logits)

        next_q_values = np.sum(next_probs * self.z, axis=2)
        best_next_actions = np.argmax(next_q_values, axis=1)

        # Select probabilities for best next actions
        batch_idx = np.arange(batch_size)
        best_next_probs = next_probs[batch_idx, best_next_actions, :]

        # Compute projection
        m = np.zeros((batch_size, self.num_atoms))

        for i in range(batch_size):
            if dones[i]:
                Tz = np.clip(rewards[i], self.v_min, self.v_max)
                b = (Tz - self.v_min) / self.delta_z
                l = int(np.floor(b))
                u = int(np.ceil(b))

                if l == u:
                    m[i, l] += 1
                else:
                    m[i, l] += (u - b)
                    m[i, u] += (b - l)
            else:
                for j in range(self.num_atoms):
                    Tz = np.clip(rewards[i] + gamma * self.z[j], self.v_min, self.v_max)
                    b = (Tz - self.v_min) / self.delta_z
                    l = int(np.floor(b))
                    u = int(np.ceil(b))

                    if l == u:
                        m[i, l] += best_next_probs[i, j]
                    else:
                        m[i, l] += best_next_probs[i, j] * (u - b)
                        m[i, u] += best_next_probs[i, j] * (b - l)

        # Cross entropy loss and gradients
        logits = np.dot(states, self.W) + self.b
        logits = logits.reshape(batch_size, self.action_dim, self.num_atoms)

        # Forward pass for taken actions
        action_logits = logits[batch_idx, actions, :]
        action_probs = self._softmax(action_logits)

        loss = -np.sum(m * np.log(action_probs + 1e-8)) / batch_size

        # Backprop
        d_action_logits = (action_probs - m) / batch_size

        d_logits = np.zeros_like(logits)
        d_logits[batch_idx, actions, :] = d_action_logits
        d_logits = d_logits.reshape(batch_size, -1)

        dW = np.dot(states.T, d_logits)
        db = np.sum(d_logits, axis=0)

        self.W -= lr * dW
        self.b -= lr * db

        return loss

    def update_target(self):
        self.target_W = np.copy(self.W)
        self.target_b = np.copy(self.b)

if __name__ == "__main__":
    np.random.seed(42)
    state_dim = 4
    action_dim = 2

    agent = C51(state_dim, action_dim)

    # Synthetic environment mapping state to optimal discrete action
    def get_reward_and_done(state, action):
        optimal = 1 if np.sum(state) > 0 else 0
        if action == optimal:
            return 1.0, 1
        return -1.0, 0

    print("Testing C51 Component...")
    for epoch in range(1000):
        states = np.random.uniform(-1, 1, size=(64, state_dim))
        actions = []
        rewards = []
        dones = []
        next_states = []

        for i in range(64):
            a = agent.get_action(states[i])
            if np.random.rand() < 0.2:
                a = np.random.randint(action_dim)
            r, d = get_reward_and_done(states[i], a)
            actions.append(a)
            rewards.append(r)
            dones.append(d)
            next_states.append(np.random.uniform(-1, 1, size=(state_dim)))

        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        next_states = np.array(next_states)

        loss = agent.train(states, actions, rewards, next_states, dones)
        if epoch % 200 == 0:
            agent.update_target()
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Evaluation
    test_state1 = np.array([0.5, 0.5, 0.5, 0.5])
    test_state2 = np.array([-0.5, -0.5, -0.5, -0.5])

    opt1 = 1
    opt2 = 0

    pred1 = agent.get_action(test_state1)
    pred2 = agent.get_action(test_state2)

    print("\nEvaluation:")
    print(f"State 1: Optimal {opt1}, Predicted {pred1}")
    print(f"State 2: Optimal {opt2}, Predicted {pred2}")

    if opt1 == pred1 and opt2 == pred2:
        print("C51 successfully learned the optimal discrete policy.")
    else:
        print("C51 failed to learn the precise policy.")
