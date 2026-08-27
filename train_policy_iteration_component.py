import numpy as np

class PolicyIteration:
    def __init__(self, n_states, n_actions, P, R, gamma=0.99, theta=1e-5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.R = R
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(n_states)
        self.policy = np.zeros(n_states, dtype=int)

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(self.n_states):
                v = self.V[s]
                a = self.policy[s]
                self.V[s] = self.R[s, a] + self.gamma * np.dot(self.P[s, a], self.V)
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in range(self.n_states):
            old_action = self.policy[s]
            q_values = self.R[s] + self.gamma * np.dot(self.P[s], self.V)
            self.policy[s] = np.argmax(q_values)
            if old_action != self.policy[s]:
                policy_stable = False
        return policy_stable

    def train(self):
        iteration = 0
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            iteration += 1
            if policy_stable:
                break

        print(f"Policy iteration converged after {iteration} iterations.")
        return self.V, self.policy

if __name__ == "__main__":
    n_states = 3
    n_actions = 2

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0

    P[1, 0, 0] = 1.0
    P[1, 1, 2] = 1.0

    P[2, 0, 2] = 1.0
    P[2, 1, 2] = 1.0

    R[1, 1] = 10.0
    R[0, 1] = 0.0

    pi = PolicyIteration(n_states, n_actions, P, R, gamma=0.9)
    V, policy = pi.train()

    print("Optimal Value Function:", V)
    print("Optimal Policy:", policy)
    assert np.all(V > -1)
    print("Optimization finished successfully.")
