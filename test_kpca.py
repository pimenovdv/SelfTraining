import numpy as np
from train_kpca_component import KernelPCA

def test_kpca():
    np.random.seed(42)
    theta1 = np.linspace(0, 2 * np.pi, 100)
    r1 = 2
    X1 = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]

    theta2 = np.linspace(0, 2 * np.pi, 100)
    r2 = 6
    X2 = np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)]

    X = np.vstack([X1, X2])
    y = np.array([0] * 100 + [1] * 100)

    X += np.random.randn(*X.shape) * 0.2

    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1)
    X_kpca = kpca.fit_transform(X)

    X_aug = np.c_[np.ones(X_kpca.shape[0]), X_kpca]
    w = np.linalg.inv(X_aug.T @ X_aug + np.eye(X_aug.shape[1]) * 1e-4) @ X_aug.T @ y
    preds = (X_aug @ w > 0.5).astype(int)
    accuracy = np.mean(preds == y)

    assert accuracy > 0.95, "KPCA failed to linearly separate concentric circles."
    print("KPCA Test Passed.")

if __name__ == "__main__":
    test_kpca()
