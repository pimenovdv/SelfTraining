import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MADDPG_Agent:
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=64, lr_actor=0.01, lr_critic=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        # Actor network (local state -> local action)
        self.W1_a = np.random.randn(state_dim, hidden_dim) * np.sqrt(2. / state_dim)
        self.b1_a = np.zeros(hidden_dim)
        self.W2_a = np.random.randn(hidden_dim, action_dim) * np.sqrt(2. / hidden_dim)
        self.b2_a = np.zeros(action_dim)

        # Centralized Critic network (all states + all actions -> Q-value)
        critic_input_dim = (state_dim * n_agents) + (action_dim * n_agents)
        self.W1_c = np.random.randn(critic_input_dim, hidden_dim) * np.sqrt(2. / critic_input_dim)
        self.b1_c = np.zeros(hidden_dim)
        self.W2_c = np.random.randn(hidden_dim, 1) * np.sqrt(2. / hidden_dim)
        self.b2_c = np.zeros(1)

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

    def actor_forward(self, state):
        self.z1_a = np.dot(state, self.W1_a) + self.b1_a
        self.a1_a = relu(self.z1_a)
        self.z2_a = np.dot(self.a1_a, self.W2_a) + self.b2_a
        # Using softmax for discrete action selection probability
        return softmax(self.z2_a)

    def critic_forward(self, joint_state, joint_action):
        x = np.concatenate((joint_state, joint_action), axis=-1)
        self.z1_c = np.dot(x, self.W1_c) + self.b1_c
        self.a1_c = relu(self.z1_c)
        self.q_value = np.dot(self.a1_c, self.W2_c) + self.b2_c
        return self.q_value

    def critic_backward_for_q_value(self, joint_state, joint_action, target_q):
        # MSE loss derivative wrt Q
        x = np.concatenate((joint_state, joint_action), axis=-1)
        dq = (self.q_value - target_q)

        dW2_c = np.dot(self.a1_c.T, dq)
        db2_c = np.sum(dq, axis=0)

        da1_c = np.dot(dq, self.W2_c.T)
        dz1_c = da1_c * relu_derivative(self.z1_c)

        dW1_c = np.dot(x.T, dz1_c)
        db1_c = np.sum(dz1_c, axis=0)

        # Update critic
        self.W1_c -= self.lr_critic * dW1_c
        self.b1_c -= self.lr_critic * db1_c
        self.W2_c -= self.lr_critic * dW2_c
        self.b2_c -= self.lr_critic * db2_c

    def critic_backward_for_actor(self, joint_state, joint_action):
        # We need dQ/da to update actor. We assume q_value was already computed.
        # dQ/dq = 1
        dq = np.ones_like(self.q_value)

        da1_c = np.dot(dq, self.W2_c.T)
        dz1_c = da1_c * relu_derivative(self.z1_c)

        dx = np.dot(dz1_c, self.W1_c.T)
        # return gradients of Q wrt joint_action
        d_joint_action = dx[:, self.state_dim * self.n_agents:]
        return d_joint_action

    def actor_backward(self, state, d_action):
        # d_action is dQ/da (from critic_backward_for_actor).
        # We want to maximize Q, so we do gradient ascent on actor

        # Simplified approximation for deterministic policy gradient with softmax
        dz2_a = d_action

        dW2_a = np.dot(self.a1_a.T, dz2_a)
        db2_a = np.sum(dz2_a, axis=0)

        da1_a = np.dot(dz2_a, self.W2_a.T)
        dz1_a = da1_a * relu_derivative(self.z1_a)

        dW1_a = np.dot(state.T, dz1_a)
        db1_a = np.sum(dz1_a, axis=0)

        # Gradient ascent
        self.W1_a += self.lr_actor * dW1_a
        self.b1_a += self.lr_actor * db1_a
        self.W2_a += self.lr_actor * dW2_a
        self.b2_a += self.lr_actor * db2_a

def train_maddpg():
    np.random.seed(42)
    n_agents = 2
    state_dim = 4
    action_dim = 2
    hidden_dim = 16
    epochs = 1000
    batch_size = 32

    agents = [MADDPG_Agent(state_dim, action_dim, n_agents, hidden_dim) for _ in range(n_agents)]

    # Synthetic dataset

    losses = []

    for epoch in range(epochs):
        # Generate batch
        states = np.random.randn(batch_size, n_agents, state_dim)

        joint_state = states.reshape(batch_size, -1)

        # Forward actors
        actions = []
        for i in range(n_agents):
            a = agents[i].actor_forward(states[:, i, :])
            actions.append(a)

        joint_action = np.concatenate(actions, axis=-1)

        # Synthetic reward: positive if agents choose different actions
        r = np.sum((actions[0] - actions[1])**2, axis=-1, keepdims=True)
        # target_q is just reward for this simple step (gamma=0)
        target_q = r

        # Train Critic and Actor
        total_loss = 0
        for i in range(n_agents):
            # Forward critic
            q = agents[i].critic_forward(joint_state, joint_action)
            total_loss += np.mean((q - target_q)**2)

            # Update Critic
            agents[i].critic_backward_for_q_value(joint_state, joint_action, target_q)

            # Get dQ/da for Actor update
            d_joint_action = agents[i].critic_backward_for_actor(joint_state, joint_action)
            d_action_i = d_joint_action[:, i*action_dim : (i+1)*action_dim]

            # Update Actor (Gradient Ascent)
            agents[i].actor_backward(states[:, i, :], d_action_i)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Critic Loss: {total_loss:.4f}")

    print("Training finished. Final Loss:", total_loss)
    assert total_loss < 0.1, "Loss did not converge"

if __name__ == "__main__":
    train_maddpg()
