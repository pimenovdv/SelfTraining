import numpy as np
from train_factor_analysis_component import FactorAnalysis

def test_factor_analysis():
    np.random.seed(42)
    n_samples = 1000
    z = np.random.randn(n_samples, 2)
    W_true = np.random.randn(5, 2)
    noise = np.random.randn(n_samples, 5) * 0.1
    X = z @ W_true.T + noise

    fa = FactorAnalysis(n_components=2)
    fa.fit(X)

    assert not np.isnan(fa.components_).any()
    assert not np.isnan(fa.noise_variance_).any()
    assert fa.components_.shape == (2, 5)

    z_pred = fa.transform(X)
    assert z_pred.shape == (1000, 2)
    print("test_factor_analysis passed")

if __name__ == "__main__":
    test_factor_analysis()
