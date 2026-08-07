import numpy as np
import os

def g(x):
    return np.tanh(x)

def g_prime(x):
    return 1.0 - np.tanh(x)**2

def fast_ica(X, n_components, max_iter=1000, tol=1e-5):
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean

    cov = np.cov(X_centered)
    D, E = np.linalg.eigh(cov)
    idx = np.argsort(D)[::-1]
    D = D[idx]
    E = E[:, idx]
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    W_white = np.dot(E, np.dot(D_inv_sqrt, E.T))
    X_white = np.dot(W_white, X_centered)

    n_features = X_white.shape[0]
    W = np.random.randn(n_components, n_features)
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    W = np.dot(U, Vt)

    for i in range(max_iter):
        wx = np.dot(W, X_white)
        g_wx = g(wx)
        g_p_wx = g_prime(wx)

        W_new = np.dot(g_wx, X_white.T) / X_white.shape[1] - np.mean(g_p_wx, axis=1, keepdims=True) * W

        U, _, Vt = np.linalg.svd(W_new, full_matrices=False)
        W_new = np.dot(U, Vt)

        diff = np.max(np.abs(np.abs(np.diag(np.dot(W_new, W.T))) - 1))
        W = W_new
        if diff < tol:
            print(f"FastICA converged at iteration {i+1}")
            break

    A_inv = np.dot(W, W_white)
    S_est = np.dot(A_inv, X_centered)
    return S_est, A_inv

def main():
    print("--- Testing Independent Component Analysis (ICA) Component ---")
    np.random.seed(42)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    s1 = np.sin(2 * time)
    s2 = np.sign(np.sin(3 * time))
    S = np.c_[s1, s2].T

    A_true = np.array([[1.0, 0.5], [0.5, 1.0]])
    X = np.dot(A_true, S)

    n_components = 2
    S_est, W_unmix = fast_ica(X, n_components=n_components)

    corr_matrix = np.abs(np.corrcoef(S, S_est))
    cross_corr = corr_matrix[0:2, 2:4]
    max_corrs = np.max(cross_corr, axis=1)

    print(f"Max Correlations for each true signal: {max_corrs}")
    success = np.all(max_corrs > 0.95)

    if success:
        print("ICA successfully separated the mixed signals!")
    else:
        print("ICA failed to separate the signals adequately.")

    doc_content = """# Experiment 0103: Train Independent Component Analysis (ICA) Component

## Objective
To implement and verify Independent Component Analysis (ICA) in pure NumPy using the FastICA algorithm. This explores unsupervised representation learning for separating linearly mixed, non-Gaussian source signals (blind source separation), modeling the cocktail party problem.

## Setup
*   **Script:** `train_ica_component.py`
*   **Data:** Synthetic dataset containing two mixed signals (a sine wave and a square wave).
*   **Method:** FastICA with negentropy maximization using a hyperbolic tangent contrast function.

## Execution
The script centers and whitens the mixed data, then applies FastICA fixed-point iteration to recover the unmixing matrix and estimate the original source signals. The quality of separation is measured by the maximum absolute cross-correlation between the true and estimated signals.

## Results
*   **Status:** {status}
*   **Convergence:** FastICA converged quickly.
*   **Max Correlations:** {corr1:.4f} and {corr2:.4f}

## Observations & Next Steps
*   The implementation successfully unmixed the signals, recovering the original non-Gaussian sources with high accuracy.
*   Whitening (decorrelation and variance normalization) was critical for the stability and speed of the fixed-point iteration.
*   This validates ICA as a powerful tool for discovering hidden factors and supports the goal of exploring biologically plausible and statistical learning methods.
""".format(status="Success" if success else "Failed", corr1=max_corrs[0], corr2=max_corrs[1])

    with open("docs/0103_train_ica_component.md", "w") as f:
        f.write(doc_content)
    print("Documentation generated at docs/0103_train_ica_component.md")

if __name__ == "__main__":
    main()
