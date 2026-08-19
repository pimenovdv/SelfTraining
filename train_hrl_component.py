import numpy as np

class HRLComponent:
    def __init__(self, state_dim, num_options, num_actions):
        self.meta_w = np.random.randn(state_dim, num_options) * 0.1
        self.low_w = np.random.randn(state_dim + num_options, num_actions) * 0.1

    def meta_policy(self, state):
        logits = np.dot(state, self.meta_w)
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def low_policy(self, state, option):
        opt_vec = np.zeros(self.meta_w.shape[1])
        opt_vec[option] = 1.0
        combined = np.concatenate([state, opt_vec])
        logits = np.dot(combined, self.low_w)
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

if __name__ == "__main__":
    np.random.seed(42)
    state_dim = 4
    num_options = 2
    num_actions = 3

    agent = HRLComponent(state_dim, num_options, num_actions)
    state = np.random.randn(state_dim)

    option_probs = agent.meta_policy(state)
    option = np.argmax(option_probs)

    action_probs = agent.low_policy(state, option)
    action = np.argmax(action_probs)

    print("Executing HRL Component (mathematical proxy)...")
    print(f"State: {np.round(state, 2)}")
    print(f"Meta-Controller Option Probs: {np.round(option_probs, 2)}")
    print(f"Selected Option: {option}")
    print(f"Controller Action Probs: {np.round(action_probs, 2)}")
    print(f"Selected Action: {action}")
    print("Success")
