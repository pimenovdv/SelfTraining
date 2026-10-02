import numpy as np
import os

def kl_divergence(p, q, epsilon=1e-10):
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))

def js_divergence(p, q, epsilon=1e-10):
    """
    Computes the Jensen-Shannon Divergence between two probability distributions.
    JSD(P || Q) = 1/2 * D_KL(P || M) + 1/2 * D_KL(Q || M)
    where M = 1/2 * (P + Q)
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, epsilon) + 0.5 * kl_divergence(q, m, epsilon)

def test_component():
    print("Testing Jensen-Shannon Divergence component...")

    # Target distribution (P)
    p = np.array([0.4, 0.4, 0.2])

    # Predicted distribution 1 (Q1 - very close to P)
    q1 = np.array([0.38, 0.42, 0.2])

    # Predicted distribution 2 (Q2 - far from P)
    q2 = np.array([0.1, 0.1, 0.8])

    js_1 = js_divergence(p, q1)
    js_2 = js_divergence(p, q2)
    js_symmetric_1 = js_divergence(q1, p)

    print(f"Target P: {p}")
    print(f"Predicted Q1 (close): {q1}")
    print(f"JS Divergence JSD(P || Q1): {js_1:.6f}")

    print(f"JS Divergence JSD(Q1 || P): {js_symmetric_1:.6f}")

    print(f"Predicted Q2 (far): {q2}")
    print(f"JS Divergence JSD(P || Q2): {js_2:.6f}")

    assert js_1 < js_2, "JS Divergence should be lower for the distribution closer to the target."
    assert js_1 >= 0, "JS Divergence must be non-negative."
    assert js_2 >= 0, "JS Divergence must be non-negative."
    assert np.isclose(js_1, js_symmetric_1), "JS Divergence must be symmetric."

    print("Jensen-Shannon Divergence component test passed.")

if __name__ == "__main__":
    os.makedirs("models/js_divergence", exist_ok=True)
    test_component()
