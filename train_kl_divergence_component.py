import numpy as np
import os

def kl_divergence(p, q, epsilon=1e-10):
    """
    Computes the Kullback-Leibler Divergence between two probability distributions.
    D_KL(P || Q) = sum(P * log(P / Q))
    """
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))

def test_component():
    print("Testing Kullback-Leibler Divergence component...")

    # Target distribution (P)
    p = np.array([0.4, 0.4, 0.2])

    # Predicted distribution 1 (Q1 - very close to P)
    q1 = np.array([0.38, 0.42, 0.2])

    # Predicted distribution 2 (Q2 - far from P)
    q2 = np.array([0.1, 0.1, 0.8])

    kl_1 = kl_divergence(p, q1)
    kl_2 = kl_divergence(p, q2)

    print(f"Target P: {p}")
    print(f"Predicted Q1 (close): {q1}")
    print(f"KL Divergence D_KL(P || Q1): {kl_1:.6f}")

    print(f"Predicted Q2 (far): {q2}")
    print(f"KL Divergence D_KL(P || Q2): {kl_2:.6f}")

    assert kl_1 < kl_2, "KL Divergence should be lower for the distribution closer to the target."
    assert kl_1 >= 0, "KL Divergence must be non-negative."
    assert kl_2 >= 0, "KL Divergence must be non-negative."

    print("KL Divergence component test passed.")

if __name__ == "__main__":
    os.makedirs("models/kl_divergence", exist_ok=True)
    test_component()
