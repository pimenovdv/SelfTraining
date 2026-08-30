import numpy as np

def run_linucb(n_arms=5, d=10, n_steps=2000, alpha=1.0):
    true_theta = np.random.randn(n_arms, d)

    A = [np.eye(d) for _ in range(n_arms)]
    b = [np.zeros(d) for _ in range(n_arms)]

    rewards = np.zeros(n_steps)
    optimal_rewards = np.zeros(n_steps)

    for t in range(n_steps):
        x = np.random.randn(d)
        x /= np.linalg.norm(x) # normalize context

        p = np.zeros(n_arms)
        for a in range(n_arms):
            A_inv = np.linalg.inv(A[a])
            theta_hat = A_inv.dot(b[a])
            p[a] = theta_hat.dot(x) + alpha * np.sqrt(x.dot(A_inv).dot(x))

        action = np.argmax(p)

        true_expected_rewards = [true_theta[a].dot(x) for a in range(n_arms)]
        optimal_action = np.argmax(true_expected_rewards)

        reward = true_expected_rewards[action] + np.random.randn() * 0.1

        A[action] += np.outer(x, x)
        b[action] += reward * x

        rewards[t] = reward
        optimal_rewards[t] = true_expected_rewards[optimal_action]

    cumulative_regret = np.cumsum(optimal_rewards - rewards)

    print("Final Cumulative Regret:", cumulative_regret[-1])

    avg_regret_early = np.mean(optimal_rewards[:100] - rewards[:100])
    avg_regret_late = np.mean(optimal_rewards[-100:] - rewards[-100:])
    print(f"Early average regret: {avg_regret_early:.4f}")
    print(f"Late average regret: {avg_regret_late:.4f}")

    assert avg_regret_late < avg_regret_early, "LinUCB did not learn to reduce regret"

if __name__ == "__main__":
    np.random.seed(42)
    print("Testing LinUCB Component...")
    run_linucb()
    print("Success!")
