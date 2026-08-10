import numpy as np

class SimpleEnv:
    def __init__(self):
        self.state = 2 # Start in middle
    def reset(self):
        self.state = 2
        return self.state
    def step(self, state, action):
        next_state = state - 1 if action == 0 else state + 1
        reward = 0
        done = False
        if next_state == 0:
            reward = -1.0
            done = True
        elif next_state == 4:
            reward = 1.0
            done = True
        return next_state, reward, done

class Node:
    def __init__(self, prior, state):
        self.prior = prior
        self.state = state
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.reward = 0
        self.is_terminal = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

def select_child(node):
    best_score = -float('inf')
    best_action = -1
    best_child = None
    for action, child in node.children.items():
        q_value = child.reward + 0.9 * child.value()
        score = q_value + 1.0 * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child

class MCTS:
    def __init__(self, nn, env, num_simulations=50):
        self.nn = nn
        self.env = env
        self.num_simulations = num_simulations

    def search(self, initial_state):
        pi, v = self.nn.forward_state(initial_state)
        root = Node(prior=1.0, state=initial_state)

        for action, prob in enumerate(pi):
            next_state, reward, done = self.env.step(initial_state, action)
            child = Node(prior=prob, state=next_state)
            child.reward = reward
            child.is_terminal = done
            root.children[action] = child

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.children:
                action, node = select_child(node)
                search_path.append(node)

            value = 0
            if not node.is_terminal:
                pi, v = self.nn.forward_state(node.state)
                value = v[0]
                for a, prob in enumerate(pi):
                    next_s, r, d = self.env.step(node.state, a)
                    child = Node(prior=prob, state=next_s)
                    child.reward = r
                    child.is_terminal = d
                    node.children[a] = child
            else:
                value = 0

            ret = value
            for n in reversed(search_path):
                n.value_sum += ret
                n.visit_count += 1
                ret = n.reward + 0.9 * ret

        action_visits = [root.children[a].visit_count for a in range(2)]
        sum_visits = sum(action_visits)
        return [v / sum_visits for v in action_visits]

class AlphaZeroNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2_pi = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)
        self.b2_pi = np.zeros((1, output_dim))
        self.W2_v = np.random.randn(hidden_dim, 1) * np.sqrt(2/hidden_dim)
        self.b2_v = np.zeros((1, 1))

    def forward_state(self, state):
        x = np.zeros((1, 5))
        x[0, state] = 1.0
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)

        self.z_pi = self.h1 @ self.W2_pi + self.b2_pi
        exp_z = np.exp(self.z_pi - np.max(self.z_pi, axis=1, keepdims=True))
        self.pi = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        self.z_v = self.h1 @ self.W2_v + self.b2_v
        self.v = np.tanh(self.z_v)

        return self.pi[0], self.v[0]

    def backward(self, target_pi, target_v, lr=0.01):
        d_z_pi = self.pi - target_pi
        d_v = self.v - target_v
        d_z_v = d_v * (1 - self.v ** 2)

        dW2_pi = self.h1.T @ d_z_pi
        db2_pi = np.sum(d_z_pi, axis=0, keepdims=True)

        dW2_v = self.h1.T @ d_z_v
        db2_v = np.sum(d_z_v, axis=0, keepdims=True)

        d_h1 = d_z_pi @ self.W2_pi.T + d_z_v @ self.W2_v.T
        d_z1 = d_h1 * (self.z1 > 0)

        dW1 = self.x.T @ d_z1
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        self.W2_pi -= lr * dW2_pi
        self.b2_pi -= lr * db2_pi
        self.W2_v -= lr * dW2_v
        self.b2_v -= lr * db2_v
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

env = SimpleEnv()
net = AlphaZeroNet(input_dim=5, hidden_dim=16, output_dim=2)

epochs = 100
for epoch in range(epochs):
    states, target_pis, rewards = [], [], []
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < 20:
        mcts = MCTS(net, env, num_simulations=20)
        pi_target = mcts.search(state)

        states.append(state)
        target_pis.append(pi_target)

        action = np.argmax(pi_target) # Use argmax for faster learning or choice for exploration
        next_state, reward, done = env.step(state, action)
        rewards.append(reward)
        state = next_state
        steps += 1

    returns = []
    ret = 0
    for r in reversed(rewards):
        ret = r + 0.9 * ret
        returns.insert(0, ret)

    for s, pi, ret in zip(states, target_pis, returns):
        x = np.zeros((1, 5))
        x[0, s] = 1.0

        t_pi = np.array([pi])
        t_v = np.array([[ret]])

        net.forward(x)
        net.backward(t_pi, t_v, lr=0.05)

print("Final evaluation:")
for s in [1, 2, 3]:
    pi, v = net.forward_state(s)
    print(f"State {s}: Pi={pi}, V={v[0]:.3f}")
