import numpy as np

def run_thompson_sampling(n_arms=3, n_steps=1000):
    # True probabilities of each arm
    true_probs = np.array([0.2, 0.5, 0.8])

    # Priors for Beta distribution (alpha, beta) for each arm
    alphas = np.ones(n_arms)
    betas = np.ones(n_arms)

    rewards = np.zeros(n_steps)
    actions = np.zeros(n_steps)

    for t in range(n_steps):
        # Sample from the Beta distribution for each arm
        sampled_theta = np.random.beta(alphas, betas)

        # Select the arm with the highest sampled value
        action = np.argmax(sampled_theta)
        actions[t] = action

        # Simulate environment reward
        reward = np.random.binomial(1, true_probs[action])
        rewards[t] = reward

        # Update priors based on reward
        if reward == 1:
            alphas[action] += 1
        else:
            betas[action] += 1

    print("True probabilities:", true_probs)
    print("Estimated probabilities:", alphas / (alphas + betas))
    print("Total reward:", np.sum(rewards))
    print("Optimal arm selected {} times".format(np.sum(actions == np.argmax(true_probs))))

    # Assertions to mathematically verify
    assert np.argmax(alphas / (alphas + betas)) == np.argmax(true_probs), "Did not find the best arm"

if __name__ == "__main__":
    np.random.seed(42)
    print("Testing Thompson Sampling...")
    run_thompson_sampling()
    print("Success!")
