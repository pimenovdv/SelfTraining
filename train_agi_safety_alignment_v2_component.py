import numpy as np

def generate_gridworld_mdp(size=5):
    n_states = size * size
    n_actions = 4 # up, down, left, right
    P = np.zeros((n_states, n_actions, n_states))

    for s in range(n_states):
        r, c = s // size, s % size

        # up
        s_up = (r - 1) * size + c if r > 0 else s
        P[s, 0, s_up] = 1.0

        # down
        s_down = (r + 1) * size + c if r < size - 1 else s
        P[s, 1, s_down] = 1.0

        # left
        s_left = r * size + (c - 1) if c > 0 else s
        P[s, 2, s_left] = 1.0

        # right
        s_right = r * size + (c + 1) if c < size - 1 else s
        P[s, 3, s_right] = 1.0

    return n_states, n_actions, P

def value_iteration_soft(P, R, gamma=0.9, epsilon=1e-5, max_iters=1000):
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)

    for i in range(max_iters):
        Q = R[:, None] + gamma * np.sum(P * V, axis=2)
        Q_max = np.max(Q, axis=1)
        V_new = Q_max + np.log(np.sum(np.exp(Q - Q_max[:, None]), axis=1))

        if np.max(np.abs(V - V_new)) < epsilon:
            V = V_new
            break
        V = V_new

    # Compute policy
    Q = R[:, None] + gamma * np.sum(P * V, axis=2)
    Q_max = np.max(Q, axis=1)
    exp_Q = np.exp(Q - Q_max[:, None])
    policy = exp_Q / np.sum(exp_Q, axis=1)[:, None]
    return policy

def compute_state_visitation_frequencies(P, policy, start_state_probs, n_steps=20):
    n_states = P.shape[0]
    mu = np.zeros((n_steps, n_states))
    mu[0] = start_state_probs

    for t in range(1, n_steps):
        for s in range(n_states):
            mu[t, s] = np.sum(mu[t-1, :] * np.sum(policy * P[:, :, s], axis=1))

    return np.sum(mu, axis=0)

def maxent_irl(P, features, expert_trajectories, epochs=100, learning_rate=0.1, gamma=0.9):
    n_states, n_features = features.shape
    theta = np.random.uniform(size=n_features)

    # Compute expert feature expectations
    mu_expert = np.zeros(n_features)
    for traj in expert_trajectories:
        for s in traj:
            mu_expert += features[s]
    mu_expert /= len(expert_trajectories)

    start_state_probs = np.zeros(n_states)
    for traj in expert_trajectories:
        start_state_probs[traj[0]] += 1
    start_state_probs /= np.sum(start_state_probs)

    for epoch in range(epochs):
        R = np.dot(features, theta)
        policy = value_iteration_soft(P, R, gamma)
        svf = compute_state_visitation_frequencies(P, policy, start_state_probs)
        mu_model = np.dot(svf, features)

        grad = mu_expert - mu_model
        theta += learning_rate * grad

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Gradient Norm: {np.linalg.norm(grad):.4f}")

    return theta, policy

def main():
    print("Training AGI Safety Alignment V2 Component (MaxEnt IRL on Gridworld)...")
    np.random.seed(42)

    n_states, n_actions, P = generate_gridworld_mdp(size=5)

    # Features: one-hot for simplicity
    features = np.eye(n_states)

    # Generate some expert trajectories (e.g., trying to reach state 24 from state 0)
    expert_trajectories = [
        [0, 1, 2, 3, 4, 9, 14, 19, 24],
        [0, 5, 10, 15, 20, 21, 22, 23, 24],
        [0, 5, 6, 7, 12, 17, 22, 23, 24]
    ]

    theta, policy = maxent_irl(P, features, expert_trajectories, epochs=50, learning_rate=0.1, gamma=0.9)

    print("Training complete.")
    print(f"Learned reward for goal state (24): {np.dot(features[24], theta):.4f}")
    print(f"Learned reward for start state (0): {np.dot(features[0], theta):.4f}")

if __name__ == '__main__':
    main()
