import numpy as np

def train_affinity_propagation_component():
    print("Testing Affinity Propagation Component...")

    np.random.seed(42)
    X1 = np.random.randn(10, 2) + np.array([5, 5])
    X2 = np.random.randn(10, 2) + np.array([-5, -5])
    X = np.vstack([X1, X2])
    N = X.shape[0]

    S = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            S[i, j] = -np.sum((X[i] - X[j]) ** 2)

    preference = np.median(S)
    np.fill_diagonal(S, preference)

    R = np.zeros((N, N))
    A = np.zeros((N, N))

    max_iter = 200
    damping = 0.9

    for iteration in range(max_iter):
        R_old = R.copy()
        A_old = A.copy()

        AS = A + S
        I = np.argmax(AS, axis=1)
        Y = AS[np.arange(N), I]

        AS_temp = AS.copy()
        AS_temp[np.arange(N), I] = -np.inf
        Y2 = np.max(AS_temp, axis=1)

        max_AS = np.zeros_like(AS)
        for i in range(N):
            for k in range(N):
                if k == I[i]:
                    max_AS[i, k] = Y2[i]
                else:
                    max_AS[i, k] = Y[i]

        R = S - max_AS
        R = damping * R_old + (1 - damping) * R

        Rp = np.maximum(R, 0)
        np.fill_diagonal(Rp, np.diag(R))
        Rp_sum = np.sum(Rp, axis=0)

        for i in range(N):
            for k in range(N):
                if i != k:
                    sum_others = Rp_sum[k] - Rp[i, k] - Rp[k, k]
                    A[i, k] = min(0, R[k, k] + sum_others)
                else:
                    A[k, k] = Rp_sum[k] - Rp[k, k]

        A = damping * A_old + (1 - damping) * A

    E = R + A
    labels = np.argmax(E, axis=1)
    exemplars = np.unique(labels)

    print(f"Identified {len(exemplars)} clusters.")
    assert len(exemplars) == 2, "Failed to identify the correct number of clusters"

    print("Affinity Propagation Component mathematical mechanism verified successfully.")
    return True

if __name__ == "__main__":
    train_affinity_propagation_component()
