import numpy as np

class ValueIteration:
    def __init__(self, n_states, n_actions, P, R, gamma=0.99, theta=1e-5):
        """
        P: Transition probabilities (n_states, n_actions, n_states)
        R: Reward function (n_states, n_actions)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.R = R
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(n_states)
        self.policy = np.zeros(n_states, dtype=int)

    def train(self):
        iteration = 0
        while True:
            delta = 0
            for s in range(self.n_states):
                v = self.V[s]
                # Q(s, a) = R(s,a) + gamma * sum(P(s'|s,a) * V(s'))
                q_values = self.R[s] + self.gamma * np.dot(self.P[s], self.V)
                self.V[s] = np.max(q_values)
                delta = max(delta, abs(v - self.V[s]))
            iteration += 1
            if delta < self.theta:
                break

        # Extract policy
        for s in range(self.n_states):
            q_values = self.R[s] + self.gamma * np.dot(self.P[s], self.V)
            self.policy[s] = np.argmax(q_values)

        print(f"Value iteration converged after {iteration} iterations.")
        return self.V, self.policy

if __name__ == "__main__":
    # Simple Gridworld example (1D)
    # 3 states: 0, 1, 2. Terminal state is 2.
    n_states = 3
    n_actions = 2 # 0: left, 1: right

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    # State 0
    P[0, 0, 0] = 1.0 # left
    P[0, 1, 1] = 1.0 # right

    # State 1
    P[1, 0, 0] = 1.0 # left
    P[1, 1, 2] = 1.0 # right

    # State 2 (Terminal)
    P[2, 0, 2] = 1.0
    P[2, 1, 2] = 1.0

    # Rewards
    R[1, 1] = 10.0 # Reward for transitioning 1 -> 2
    R[0, 1] = 0.0

    val_iter = ValueIteration(n_states, n_actions, P, R, gamma=0.9)
    V, policy = val_iter.train()

    print("Optimal Value Function:", V)
    print("Optimal Policy:", policy)
    assert np.all(V > -1)
    print("Optimization finished successfully.")
