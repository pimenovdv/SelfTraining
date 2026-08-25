import numpy as np

def target_distribution(x):
    # A simple bimodal Gaussian mixture target distribution
    return 0.3 * np.exp(-0.5 * ((x - 2) / 0.5)**2) + 0.7 * np.exp(-0.5 * ((x + 2) / 0.5)**2)

def metropolis_hastings(target_pdf, num_samples, initial_state=0.0, proposal_std=1.0):
    samples = np.zeros(num_samples)
    current_state = initial_state

    for i in range(num_samples):
        # Propose a new state from a Gaussian proposal distribution
        proposed_state = np.random.normal(current_state, proposal_std)

        # Calculate acceptance probability
        p_current = target_pdf(current_state)
        p_proposed = target_pdf(proposed_state)

        # Acceptance ratio (symmetric proposal, so q(x'|x) = q(x|x'))
        acceptance_ratio = p_proposed / p_current if p_current > 0 else 1.0

        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state

        samples[i] = current_state

    return samples

if __name__ == "__main__":
    np.random.seed(42)
    print("Initializing MCMC (Metropolis-Hastings) component testing...")

    num_samples = 10000
    samples = metropolis_hastings(target_distribution, num_samples, initial_state=0.0, proposal_std=2.0)

    # Calculate sample mean and variance
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)

    print(f"Generated {num_samples} samples.")
    print(f"Sample Mean: {sample_mean:.4f}")
    print(f"Sample Variance: {sample_var:.4f}")

    # Check if mean and variance are somewhat reasonable for the given mixture
    # True mean = 0.3 * 2 + 0.7 * (-2) = 0.6 - 1.4 = -0.8
    expected_mean = -0.8
    if np.abs(sample_mean - expected_mean) < 0.2:
        print("MCMC component successfully sampled from the target distribution.")
    else:
        print(f"Warning: sample mean {sample_mean:.4f} is far from expected {expected_mean:.4f}.")

    print("Testing completed.")
