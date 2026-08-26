import numpy as np
import scipy.spatial.distance as dist

def sinkhorn_knopp(M, r, c, reg, max_iter=1000, tol=1e-9):
    K = np.exp(-M / reg)
    u = np.ones(len(r)) / len(r)
    v = np.ones(len(c)) / len(c)

    for i in range(max_iter):
        u_prev = u.copy()
        v = c / (np.dot(K.T, u) + 1e-15)
        u = r / (np.dot(K, v) + 1e-15)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    P = np.diag(u) @ K @ np.diag(v)
    return P

def test_optimal_transport():
    print("Testing Optimal Transport Component...")
    np.random.seed(42)
    X = np.random.randn(10, 2)
    Y = np.random.randn(10, 2) + np.array([2, 2])
    M = dist.cdist(X, Y, metric='sqeuclidean')
    M = M / M.max()
    r = np.ones(10) / 10
    c = np.ones(10) / 10
    reg = 0.05
    P = sinkhorn_knopp(M, r, c, reg)
    assert np.allclose(np.sum(P, axis=1), r, atol=1e-3), f"Row sums do not match source marginals."
    assert np.allclose(np.sum(P, axis=0), c, atol=1e-3), f"Column sums do not match target marginals."
    print("Optimal Transport testing successful!")

if __name__ == "__main__":
    test_optimal_transport()
