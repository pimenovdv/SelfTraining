import numpy as np
import time

def metropolis_hastings(target_pdf, proposal_std, initial_state, num_samples):
    samples = []
    current_state = initial_state
    current_prob = target_pdf(current_state)

    accepted = 0
    for _ in range(num_samples):
        # Propose a new state from a Gaussian distribution centered at current_state
        proposed_state = np.random.normal(current_state, proposal_std)
        proposed_prob = target_pdf(proposed_state)

        # Calculate acceptance probability
        # Since proposal distribution (Gaussian) is symmetric, q(x'|x) = q(x|x')
        # Acceptance ratio alpha = min(1, P(x') / P(x))
        if current_prob > 0:
            acceptance_ratio = proposed_prob / current_prob
        else:
            acceptance_ratio = 1.0 if proposed_prob > 0 else 0.0

        # Accept or reject
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
            current_prob = proposed_prob
            accepted += 1

        samples.append(current_state)

    return np.array(samples), accepted / num_samples

def evaluate_mcmc():
    print("Evaluating Markov Chain Monte Carlo (MCMC) via Metropolis-Hastings component...")

    # Define a complex target distribution (e.g., a mixture of two Gaussians)
    def target_distribution(x):
        return 0.3 * np.exp(-0.5 * ((x - (-2)) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi)) + \
               0.7 * np.exp(-0.5 * ((x - 3) / 1.0)**2) / (1.0 * np.sqrt(2 * np.pi))

    np.random.seed(42)
    num_samples = 50000
    burn_in = 5000
    proposal_std = 3.0 # larger proposal_std for better mixing
    initial_state = 0.0

    start_time = time.time()

    samples, acceptance_rate = metropolis_hastings(
        target_distribution,
        proposal_std,
        initial_state,
        num_samples
    )

    # Discard burn-in samples
    valid_samples = samples[burn_in:]

    end_time = time.time()

    # Calculate empirical mean and variance
    empirical_mean = np.mean(valid_samples)
    empirical_var = np.var(valid_samples)

    # True theoretical mean and variance of the mixture distribution
    true_mean = 0.3 * (-2) + 0.7 * 3
    true_var = 0.3 * (0.5**2 + (-2)**2) + 0.7 * (1.0**2 + 3**2) - true_mean**2

    mean_error = abs(empirical_mean - true_mean)
    var_error = abs(empirical_var - true_var)

    print(f"MCMC Sampling completed in {end_time - start_time:.4f} seconds")
    print(f"Acceptance Rate: {acceptance_rate:.4f} (Ideal is ~0.234 - 0.5)")
    print(f"Empirical Mean: {empirical_mean:.4f}, True Mean: {true_mean:.4f}")
    print(f"Empirical Variance: {empirical_var:.4f}, True Variance: {true_var:.4f}")

    if mean_error < 0.2 and var_error < 0.5:
        print("MCMC Component mathematically verified successfully.")
    else:
        print("MCMC Component failed mathematical verification.")
        print(f"Mean Error: {mean_error:.4f}, Variance Error: {var_error:.4f}")

if __name__ == "__main__":
    evaluate_mcmc()
