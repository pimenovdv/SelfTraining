import numpy as np

def sinkhorn_knopp(C, reg, num_iters=100):
    """
    Computes the Sinkhorn-Knopp algorithm for optimal transport.
    C: cost matrix
    reg: regularization parameter (entropy)
    """
    K = np.exp(-C / reg)
    n, m = C.shape

    # We assume uniform marginals for simplicity
    a = np.ones(n) / n
    b = np.ones(m) / m

    u = np.ones(n) / n
    v = np.ones(m) / m

    for _ in range(num_iters):
        u = a / (K @ v)
        v = b / (K.T @ u)

    P = np.diag(u) @ K @ np.diag(v)
    return P

def test_component():
    print("Testing Sinkhorn Optimal Transport component...")

    # Cost matrix
    C = np.array([[0.1, 1.0], [1.0, 0.1]])

    # Regularization
    reg = 0.05

    P = sinkhorn_knopp(C, reg)

    print(f"Cost matrix:\n{C}")
    print(f"Optimal transport plan:\n{P}")

    # Check if margins match
    margin_a = np.sum(P, axis=1)
    margin_b = np.sum(P, axis=0)

    print(f"Margin A: {margin_a}")
    print(f"Margin B: {margin_b}")

    assert np.allclose(margin_a, [0.5, 0.5]), "Margin A does not match uniform distribution."
    assert np.allclose(margin_b, [0.5, 0.5]), "Margin B does not match uniform distribution."

    print("Sinkhorn Optimal Transport component test passed.")

if __name__ == "__main__":
    test_component()
