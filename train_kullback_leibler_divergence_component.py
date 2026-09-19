import numpy as np

def kl_divergence(p, q, epsilon=1e-12):
    """
    Computes the Kullback-Leibler Divergence between two probability distributions p and q.
    D_KL(P || Q) = sum(P * log(P / Q))
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    # Normalize to ensure they are true probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sum(p * np.log(p / q))

def test_kullback_leibler_divergence_component():
    print("Testing Kullback-Leibler Divergence Component...")

    # Example 1: Identical distributions (KL should be 0)
    p1 = np.array([0.2, 0.3, 0.5])
    q1 = np.array([0.2, 0.3, 0.5])
    kl_1 = kl_divergence(p1, q1)
    print(f"Test 1 - Identical distributions: D_KL(P||Q) = {kl_1:.4f} (Expected: 0.0000)")
    assert np.isclose(kl_1, 0.0, atol=1e-5), "Test 1 Failed"

    # Example 2: Different distributions
    p2 = np.array([0.5, 0.3, 0.2])
    q2 = np.array([0.1, 0.4, 0.5])
    kl_2 = kl_divergence(p2, q2)
    print(f"Test 2 - Different distributions: D_KL(P||Q) = {kl_2:.4f}")
    assert kl_2 > 0.0, "Test 2 Failed"

    # Example 3: Edge cases (zeros in distribution)
    p3 = np.array([1.0, 0.0, 0.0])
    q3 = np.array([0.1, 0.8, 0.1])
    kl_3 = kl_divergence(p3, q3)
    print(f"Test 3 - Edge cases (zeros in P): D_KL(P||Q) = {kl_3:.4f}")
    assert kl_3 > 0.0, "Test 3 Failed"

    print("Kullback-Leibler Divergence Component tests passed successfully.")

if __name__ == "__main__":
    test_kullback_leibler_divergence_component()
