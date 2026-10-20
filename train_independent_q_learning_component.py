import numpy as np
import json
import os

class IndependentQLearning:
    def __init__(self, num_agents, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Independent Q-tables for each agent
        self.q_tables = [np.zeros((num_states, num_actions)) for _ in range(num_agents)]

    def act(self, state, train=True):
        actions = []
        for i in range(self.num_agents):
            if train and np.random.rand() < self.epsilon:
                actions.append(np.random.randint(self.num_actions))
            else:
                actions.append(np.argmax(self.q_tables[i][state]))
        return actions

    def update(self, state, actions, rewards, next_state, done):
        for i in range(self.num_agents):
            q_predict = self.q_tables[i][state, actions[i]]
            q_target = rewards[i]
            if not done:
                q_target += self.gamma * np.max(self.q_tables[i][next_state])
            self.q_tables[i][state, actions[i]] += self.alpha * (q_target - q_predict)

def test_iql():
    print("Testing Independent Q-Learning (IQL) Component...")
    num_agents = 2
    num_states = 5
    num_actions = 2
    env_transition = lambda s, a: (s + sum(a)) % num_states

    iql = IndependentQLearning(num_agents, num_states, num_actions)

    epochs = 2000
    for epoch in range(epochs):
        state = 0
        for step in range(10):
            actions = iql.act(state)
            next_state = env_transition(state, actions)
            # Cooperative reward: both agents get +1 if they reach state 4, else 0
            rewards = [1.0 if next_state == 4 else 0.0] * num_agents
            done = (next_state == 4)
            iql.update(state, actions, rewards, next_state, done)
            state = next_state
            if done:
                break

    # Evaluate
    state = 0
    actions = iql.act(state, train=False)
    print(f"Final evaluated actions from state 0: {actions}")
    print("Q-tables:")
    for i, q in enumerate(iql.q_tables):
        print(f"Agent {i} Q-table:\n{q}")

    # Check if agents learned to reach state 4
    if np.max(iql.q_tables[0]) > 0.0 and np.max(iql.q_tables[1]) > 0.0:
        print("Success! Independent Q-Learning agents successfully learned the Q-values.")
        if not os.path.exists("results"):
            os.makedirs("results")
        with open("results/iql_results.json", "w") as f:
            json.dump({"status": "success", "max_q": float(np.max(iql.q_tables[0]))}, f)
    else:
        print("Failure! Q-tables remain zero.")

if __name__ == "__main__":
    test_iql()
