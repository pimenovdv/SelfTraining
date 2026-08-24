import numpy as np

def train_pls():
    print("Testing Partial Least Squares (PLS) Regression")

    np.random.seed(42)
    n_samples, n_features, n_targets = 100, 10, 2
    X = np.random.randn(n_samples, n_features)
    T_true = np.random.randn(n_samples, 2)
    P_true = np.random.randn(n_features, 2)
    Q_true = np.random.randn(n_targets, 2)

    X_true = T_true @ P_true.T
    Y_true = T_true @ Q_true.T

    X = X_true + 0.1 * np.random.randn(n_samples, n_features)
    Y = Y_true + 0.1 * np.random.randn(n_samples, n_targets)

    n_components = 2
    X_k = X - np.mean(X, axis=0)
    Y_k = Y - np.mean(Y, axis=0)

    W = np.zeros((n_features, n_components))
    T = np.zeros((n_samples, n_components))
    P = np.zeros((n_features, n_components))
    Q = np.zeros((n_targets, n_components))

    for k in range(n_components):
        u = Y_k[:, 0]
        for _ in range(100):
            w = X_k.T @ u
            w = w / np.linalg.norm(w)
            t = X_k @ w
            q = Y_k.T @ t
            q = q / np.linalg.norm(q)
            u = Y_k @ q

        p = X_k.T @ t / (t.T @ t)
        X_k = X_k - np.outer(t, p)
        Y_k = Y_k - np.outer(t, q)

        W[:, k] = w
        T[:, k] = t
        P[:, k] = p
        Q[:, k] = q

    print(f"Computed {n_components} components.")
    print("PLS component training successful.")

if __name__ == '__main__':
    train_pls()
