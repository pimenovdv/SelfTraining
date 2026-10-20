import numpy as np

class InverseReinforcementLearning:
    """
    Inverse Reinforcement Learning (IRL) - Maximum Entropy IRL Proxy

    This component tests the mathematical framework for recovering a reward function
    from expert demonstrations. The agent learns the underlying reward function
    that makes the expert's behavior optimal by matching feature expectations.
    """
    def __init__(self, n_states, n_features):
        self.n_states = n_states
        self.n_features = n_features
        # Reward weights initialized randomly
        self.theta = np.random.uniform(size=(n_features,))

    def get_rewards(self, state_features):
        # Linear reward function: R(s) = theta^T * phi(s)
        return np.dot(state_features, self.theta)

    def expected_state_visitations(self, transition_probs, rewards, n_steps=20):
        # Softmax Value Iteration (per state)
        V = np.zeros(self.n_states)

        for _ in range(n_steps):
            # Q(s, s') = r(s) + gamma * V(s')
            # For this simple formulation, we assume actions correspond directly to target states
            Q = np.zeros((self.n_states, self.n_states))
            for s in range(self.n_states):
                for s_prime in range(self.n_states):
                    Q[s, s_prime] = rewards[s] + 0.9 * V[s_prime]

            # V(s) = log sum exp (Q(s, a))
            max_q = np.max(Q, axis=1)
            V = max_q + np.log(np.sum(np.exp(Q - max_q[:, None]), axis=1))

        # Compute policy pi(a|s)
        policy = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            q = rewards[s] + 0.9 * V
            policy[s] = np.exp(q - V[s])
            policy[s] /= np.sum(policy[s])

        # Compute expected state visitations (forward pass)
        visitations = np.ones(self.n_states) / self.n_states

        # transition matrix under policy
        P_pi = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            for s_prime in range(self.n_states):
                P_pi[s, s_prime] = policy[s, s_prime] * transition_probs[s, s_prime]

        # Normalize rows to ensure valid probability transitions
        row_sums = np.sum(P_pi, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 # avoid div by zero
        P_pi /= row_sums

        for _ in range(20):
            visitations = np.dot(visitations, P_pi)

        return visitations

    def train_step(self, expert_feature_expectations, state_features, transition_probs, lr=0.1):
        rewards = self.get_rewards(state_features)

        # Calculate expected feature counts under current reward
        visitations = self.expected_state_visitations(transition_probs, rewards)
        expected_features = np.dot(visitations, state_features)

        # Gradient of MaxEnt IRL objective: Expert Features - Expected Features
        grad = expert_feature_expectations - expected_features

        # Update weights (gradient ascent to maximize likelihood)
        self.theta += lr * grad

        # L2 Regularization
        self.theta -= lr * 0.01 * self.theta

        return np.linalg.norm(grad)

def test_irl():
    print("Testing Inverse Reinforcement Learning (IRL) Component...")
    np.random.seed(42)

    n_states = 10
    n_features = 3

    state_features = np.random.rand(n_states, n_features)
    transition_probs = np.random.rand(n_states, n_states)
    transition_probs = transition_probs / transition_probs.sum(axis=1, keepdims=True)

    # Define a true underlying reward structure (expert)
    true_theta = np.array([0.8, -0.4, 0.5])

    # We create a dummy instance just to calculate true expert visitations
    expert_irl = InverseReinforcementLearning(n_states, n_features)
    expert_rewards = np.dot(state_features, true_theta)
    expert_visitations = expert_irl.expected_state_visitations(transition_probs, expert_rewards)
    expert_feature_expectations = np.dot(expert_visitations, state_features)

    irl = InverseReinforcementLearning(n_states, n_features)

    epochs = 200
    print(f"Training MaxEnt IRL proxy for {epochs} epochs...")

    for epoch in range(epochs):
        grad_norm = irl.train_step(expert_feature_expectations, state_features, transition_probs, lr=0.5)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Gradient Norm: {grad_norm:.4f} | Weights: {irl.theta}")

    print("\n--- Final Results ---")
    print(f"True Weights: {true_theta}")
    print(f"Recovered Weights: {irl.theta}")

    if grad_norm < 0.05:
        print("Success! The IRL component successfully minimized the gradient norm, dynamically aligning feature expectations via Value Iteration.")
    else:
        print("Failure. The IRL component failed to converge.")
        exit(1)

if __name__ == "__main__":
    test_irl()
