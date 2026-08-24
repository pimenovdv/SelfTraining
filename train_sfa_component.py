import numpy as np

class SlowFeatureAnalysis:
    """
    Slow Feature Analysis (SFA)
    Extracts features from temporal data that vary as slowly as possible.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.W_white = None
        self.W_sfa = None
        self.W_final = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        Cov = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(Cov)
        self.W_white = np.diag(1.0 / np.sqrt(eigvals + 1e-8)) @ eigvecs.T
        Z = X_centered @ self.W_white.T
        Z_diff = Z[1:] - Z[:-1]
        Cov_diff = np.cov(Z_diff, rowvar=False)
        eigvals_diff, eigvecs_diff = np.linalg.eigh(Cov_diff)
        idx = np.argsort(eigvals_diff)
        self.W_sfa = eigvecs_diff[:, idx[:self.n_components]]
        self.W_final = self.W_white.T @ self.W_sfa

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.W_final

def test_sfa():
    print("Testing Slow Feature Analysis Component...")
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    s1 = np.sin(t)
    s2 = np.sin(10 * t)
    X = np.column_stack((s1, s2))
    A = np.array([[0.5, 0.8], [0.8, 0.3]])
    X_mixed = X @ A
    sfa = SlowFeatureAnalysis(n_components=2)
    sfa.fit(X_mixed)
    Y = sfa.transform(X_mixed)
    diff_var = np.var(Y[1:] - Y[:-1], axis=0)
    print("Variance of differences (should be ordered ascending):", diff_var)
    assert diff_var[0] < diff_var[1], "Features are not ordered by slowness."
    assert Y.shape == (1000, 2), "Transformed shape mismatch."
    print("SFA Component execution successful!")

if __name__ == "__main__":
    test_sfa()
