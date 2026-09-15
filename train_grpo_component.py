import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class GRPOComponent:
    def __init__(self, d_model, num_actions, epsilon=0.2, beta=0.01):
        self.d_model = d_model
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.beta = beta

        # Initialize policy and reference weights
        self.W_policy = np.random.randn(d_model, num_actions) * 0.1
        self.W_ref = self.W_policy.copy()

    def train_step(self, state, lr, G=8):
        # Forward pass
        logits_policy = np.dot(state, self.W_policy)
        pi_policy = softmax(logits_policy)[0]

        logits_ref = np.dot(state, self.W_ref)
        pi_ref = softmax(logits_ref)[0]

        # Sample G actions from current policy
        actions = np.random.choice(self.num_actions, size=G, p=pi_policy)

        # Simulated environment reward: action 1 is optimal
        rewards = np.array([1.0 if a == 1 else 0.0 for a in actions])

        # Compute Advantages using Group Relative normalization
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        if std_r == 0:
            std_r = 1e-8
        advantages = (rewards - mean_r) / std_r

        pi_a_policy = pi_policy[actions]
        pi_a_ref = pi_ref[actions]
        ratio = pi_a_policy / pi_a_ref

        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages

        kl = np.sum(pi_policy * np.log(pi_policy / pi_ref + 1e-10))
        loss_policy = -np.mean(np.minimum(surr1, surr2)) + self.beta * kl

        # Backpropagation
        dW = np.zeros_like(self.W_policy)
        for i in range(G):
            a = actions[i]
            adv = advantages[i]
            r_a = ratio[i]

            is_clipped = (adv > 0 and r_a > 1 + self.epsilon) or (adv < 0 and r_a < 1 - self.epsilon)
            if not is_clipped:
                one_hot = np.zeros(self.num_actions)
                one_hot[a] = 1.0
                dz = -adv * r_a * (one_hot - pi_policy)
            else:
                dz = np.zeros(self.num_actions)

            dW += np.outer(state[0], dz) / G

        dkl_dz = pi_policy * (np.log(pi_policy / pi_ref + 1e-10) - kl)
        dW += self.beta * np.outer(state[0], dkl_dz)

        self.W_policy -= lr * dW
        return loss_policy, pi_policy[1]

def test_component():
    np.random.seed(42)
    d_model = 4
    num_actions = 2

    grpo = GRPOComponent(d_model, num_actions)
    state = np.random.randn(1, d_model)

    epochs = 100
    lr = 0.1

    print("Training GRPO Component...")
    for epoch in range(epochs):
        loss, prob_opt = grpo.train_step(state, lr)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Prob Optimal Action = {prob_opt:.4f}")

    print("Success")

if __name__ == "__main__":
    test_component()
