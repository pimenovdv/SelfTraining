import numpy as np

def robust_pca(M, lam=None, tol=1e-7, max_iter=1000):
    """
    Robust PCA using Inexact ALM (Alternating Direction Method of Multipliers).
    Decomposes M = L + S where L is low rank and S is sparse.
    """
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    # Initialize
    Y = M / np.linalg.norm(M, 2)
    mu = 1.25 / np.linalg.norm(M, 2)
    rho = 1.5

    L = np.zeros((m, n))
    S = np.zeros((m, n))

    for i in range(max_iter):
        # Update S: soft thresholding
        temp_S = M - L + Y / mu
        S = np.sign(temp_S) * np.maximum(np.abs(temp_S) - lam / mu, 0)

        # Update L: singular value thresholding
        temp_L = M - S + Y / mu
        U, s, V = np.linalg.svd(temp_L, full_matrices=False)
        svp = (s > 1 / mu).sum()
        if svp >= 1:
            s_thresh = s[:svp] - 1 / mu
            L = U[:, :svp] @ np.diag(s_thresh) @ V[:svp, :]
        else:
            L = np.zeros((m, n))

        # Update Y and mu
        Z = M - L - S
        Y = Y + mu * Z
        mu = min(mu * rho, mu * 1e7)

        err = np.linalg.norm(Z, 'fro') / np.linalg.norm(M, 'fro')
        if err < tol:
            print(f"Converged at iteration {i} with error {err}")
            break

    return L, S

if __name__ == "__main__":
    print("Testing Robust PCA Component...")
    np.random.seed(42)
    m, n = 20, 20
    r = 2
    # Create low rank matrix
    U_true = np.random.randn(m, r)
    V_true = np.random.randn(r, n)
    L_true = U_true @ V_true

    # Create sparse matrix
    S_true = np.zeros((m, n))
    num_sparse = int(0.1 * m * n)
    indices = np.random.choice(m * n, num_sparse, replace=False)
    S_true.flat[indices] = 10 * np.random.randn(num_sparse)

    # Create observation matrix
    M = L_true + S_true

    L, S = robust_pca(M)

    err_L = np.linalg.norm(L - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
    print(f"Relative error of L: {err_L:.6f}")

    if err_L < 1e-2:
        print("Robust PCA implementation successful!")
    else:
        print("Robust PCA implementation failed.")
        exit(1)
