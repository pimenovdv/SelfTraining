import numpy as np
from train_probabilistic_pca_component import ProbabilisticPCA

def test_ppca():
    np.random.seed(42)
    n_samples = 1000
    z = np.random.randn(n_samples, 2)
    W_true = np.random.randn(5, 2)
    noise = np.random.randn(n_samples, 5) * 0.1
    X = z @ W_true.T + noise

    ppca = ProbabilisticPCA(n_components=2)
    ppca.fit(X)

    assert not np.isnan(ppca.components_).any()
    assert not np.isnan(ppca.noise_variance_)
    assert ppca.components_.shape == (5, 2)

    z_pred = ppca.transform(X)
    assert z_pred.shape == (1000, 2)
    print("test_ppca passed")

if __name__ == "__main__":
    test_ppca()
