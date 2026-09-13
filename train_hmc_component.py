import numpy as np

def U(q):
    return 0.5 * q**2

def grad_U(q):
    return q

def hamiltonian_monte_carlo(U, grad_U, epsilon, L, current_q):
    q = current_q
    p = np.random.randn()
    current_p = p

    p = p - epsilon * grad_U(q) / 2

    for i in range(L):
        q = q + epsilon * p
        if i != L - 1:
            p = p - epsilon * grad_U(q)

    p = p - epsilon * grad_U(q) / 2
    p = -p

    current_U = U(current_q)
    current_K = current_p**2 / 2
    proposed_U = U(q)
    proposed_K = p**2 / 2

    if np.random.rand() < np.exp(current_U - proposed_U + current_K - proposed_K):
        return q
    else:
        return current_q

def main():
    print("Testing Hamiltonian Monte Carlo (HMC) component...")
    np.random.seed(42)

    samples = []
    current_q = 0.0
    epsilon = 0.1
    L = 10
    n_samples = 10000

    for _ in range(n_samples):
        current_q = hamiltonian_monte_carlo(U, grad_U, epsilon, L, current_q)
        samples.append(current_q)

    samples = np.array(samples)
    mean = np.mean(samples)
    variance = np.var(samples)

    print(f"Sample mean: {mean:.4f}")
    print(f"Sample variance: {variance:.4f}")

    if np.abs(mean) < 0.1 and np.abs(variance - 1.0) < 0.2:
        print("HMC successfully sampled from the target distribution!")
    else:
        print("HMC failed to sample accurately.")
        exit(1)

if __name__ == "__main__":
    main()
