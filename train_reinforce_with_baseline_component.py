import numpy as np

def train():
    np.random.seed(42)

    # REINFORCE with Baseline Component
    # Adds a state-value baseline to the REINFORCE algorithm to reduce variance.

    input_size = 4
    hidden_size = 16
    output_size = 2 # 2 discrete actions

    # Policy Network (Actor)
    W1_pi = np.random.randn(input_size, hidden_size) * 0.1
    b1_pi = np.zeros(hidden_size)
    W2_pi = np.random.randn(hidden_size, output_size) * 0.1
    b2_pi = np.zeros(output_size)

    # Value Network (Critic/Baseline)
    W1_v = np.random.randn(input_size, hidden_size) * 0.1
    b1_v = np.zeros(hidden_size)
    W2_v = np.random.randn(hidden_size, 1) * 0.1
    b2_v = np.zeros(1)

    lr_pi = 0.01
    lr_v = 0.01
    gamma = 0.99
    episodes = 500

    def get_action_probs(state):
        z1 = state @ W1_pi + b1_pi
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2_pi + b2_pi
        # Softmax
        exp_z = np.exp(z2 - np.max(z2))
        return exp_z / np.sum(exp_z), a1

    def get_value(state):
        z1 = state @ W1_v + b1_v
        a1 = np.maximum(0, z1)
        v = a1 @ W2_v + b2_v
        return v[0], a1

    # Simple environment: CartPole-like (try to keep state near 0)
    def env_step(state, action):
        next_state = state + (action - 0.5) * 0.5 + np.random.randn(input_size) * 0.1
        reward = -np.sum(next_state**2) # Reward is negative distance from origin
        done = np.sum(next_state**2) > 10.0
        return next_state, reward, done

    for episode in range(episodes):
        state = np.random.randn(input_size) * 0.5
        states, actions, rewards, pi_hidden, v_hidden = [], [], [], [], []

        # Generate episode
        for t in range(50): # Max steps
            probs, h_pi = get_action_probs(state)
            action = np.random.choice(output_size, p=probs)
            _, h_v = get_value(state)

            states.append(state)
            actions.append(action)
            pi_hidden.append(h_pi)
            v_hidden.append(h_v)

            next_state, reward, done = env_step(state, action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # Calculate returns
        G = np.zeros(len(rewards))
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            G[t] = running_add

        # Normalize returns
        if len(G) > 1:
            G = (G - np.mean(G)) / (np.std(G) + 1e-8)

        # Update networks
        for t in range(len(rewards)):
            s = states[t]
            a = actions[t]
            g = G[t]
            h_pi = pi_hidden[t]
            h_v = v_hidden[t]

            v_val, _ = get_value(s)
            advantage = g - v_val

            # Value Network Update (MSE Loss)
            dv_out = -2 * advantage # Gradient of MSE: (v_val - g)^2 -> 2(v_val - g) = -2(g - v_val)
            dW2_v = np.outer(h_v, dv_out)
            db2_v = dv_out
            dh_v = dv_out * W2_v.T
            dz1_v = dh_v * (h_v > 0)
            dW1_v = np.outer(s, dz1_v)
            db1_v = dz1_v

            W1_v -= lr_v * dW1_v
            b1_v -= lr_v * db1_v[0]
            W2_v -= lr_v * dW2_v
            b2_v -= lr_v * db2_v

            # Policy Network Update
            probs, _ = get_action_probs(s)
            dlog_pi = np.copy(probs)
            dlog_pi[a] -= 1 # Gradient of log softmax

            # Scale by advantage
            dz2_pi = dlog_pi * advantage
            dW2_pi = np.outer(h_pi, dz2_pi)
            db2_pi = dz2_pi
            dh_pi = dz2_pi @ W2_pi.T
            dz1_pi = dh_pi * (h_pi > 0)
            dW1_pi = np.outer(s, dz1_pi)
            db1_pi = dz1_pi

            # We are doing gradient ASCENT on policy
            W1_pi -= lr_pi * dW1_pi
            b1_pi -= lr_pi * db1_pi
            W2_pi -= lr_pi * dW2_pi
            b2_pi -= lr_pi * db2_pi

    # Evaluate
    state = np.zeros(input_size)
    eval_rewards = []
    for _ in range(50):
        probs, _ = get_action_probs(state)
        action = np.argmax(probs) # Deterministic for evaluation
        next_state, reward, done = env_step(state, action)
        eval_rewards.append(reward)
        state = next_state
        if done: break

    final_reward = np.mean(eval_rewards)
    print(f"Final evaluation reward: {final_reward:.4f}")
    assert final_reward > -5.0, "REINFORCE with Baseline failed to learn"
    print("Success")

if __name__ == "__main__":
    train()
