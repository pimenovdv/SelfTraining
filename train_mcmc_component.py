import numpy as np

def target_pdf(x):
    # A simple bimodal target distribution (mixture of two Gaussians)
    return 0.3 * np.exp(-0.5 * ((x - (-2)) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi)) + \
           0.7 * np.exp(-0.5 * ((x - 2) / 1.0)**2) / (1.0 * np.sqrt(2 * np.pi))

def metropolis_hastings(num_samples, initial_state, proposal_std):
    samples = np.zeros(num_samples)
    current_state = initial_state

    for i in range(num_samples):
        # Propose a new state from a Gaussian proposal distribution
        proposed_state = np.random.normal(current_state, proposal_std)

        # Calculate acceptance probability
        p_accept = min(1, target_pdf(proposed_state) / target_pdf(current_state))

        # Accept or reject
        if np.random.rand() < p_accept:
            current_state = proposed_state

        samples[i] = current_state

    return samples

def main():
    print("--- Testing Markov Chain Monte Carlo (MCMC) Component ---")
    np.random.seed(42)

    num_samples = 20000
    burn_in = 5000
    initial_state = 0.0
    proposal_std = 1.0

    samples = metropolis_hastings(num_samples, initial_state, proposal_std)

    # Discard burn-in samples
    valid_samples = samples[burn_in:]

    empirical_mean = np.mean(valid_samples)
    empirical_var = np.var(valid_samples)

    print(f"Empirical Mean: {empirical_mean:.4f} (Expected: 0.8)")
    print(f"Empirical Variance: {empirical_var:.4f} (Expected: 4.135)")

    if abs(empirical_mean - 0.8) < 0.2 and abs(empirical_var - 4.135) < 0.5:
        print("Success: MCMC (Metropolis-Hastings) component verified successfully.")
    else:
        print("Failure: Sample moments significantly differ from theoretical expectations.")

if __name__ == "__main__":
    main()
