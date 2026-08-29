import numpy as np

class KernelPCA:
    def __init__(self, n_components, kernel="rbf", gamma=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.X_fit = None
        self.alphas = None
        self.lambdas = None

    def _compute_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        if self.kernel == "rbf":
            X_norm = np.sum(X ** 2, axis=-1)
            if Y is X:
                Y_norm = X_norm
            else:
                Y_norm = np.sum(Y ** 2, axis=-1)
            K = np.exp(-self.gamma * (X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)))
            return K
        elif self.kernel == "linear":
            return np.dot(X, Y.T)
        else:
            raise ValueError("Unsupported kernel")

    def fit_transform(self, X):
        self.X_fit = X
        N = X.shape[0]

        K = self._compute_kernel(X)

        one_n = np.ones((N, N)) / N
        K_c = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        eigenvalues, eigenvectors = np.linalg.eigh(K_c)

        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        self.lambdas = eigenvalues[:self.n_components]
        self.alphas = eigenvectors[:, :self.n_components]

        # Proper normalization for projection
        self.alphas = self.alphas / np.sqrt(np.maximum(self.lambdas, 1e-10))

        return np.dot(K_c, self.alphas)

if __name__ == "__main__":
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

    gamma = 0.1
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=gamma)
    X_kpca = kpca.fit_transform(X)

    X_aug = np.c_[np.ones(X_kpca.shape[0]), X_kpca]
    w = np.linalg.inv(X_aug.T @ X_aug + np.eye(X_aug.shape[1]) * 1e-4) @ X_aug.T @ y
    preds = (X_aug @ w > 0.5).astype(int)
    accuracy = np.mean(preds == y)

    print(f"Gamma = {gamma}: Linear classification accuracy in KPCA space: {accuracy * 100:.2f}%")

    if accuracy > 0.95:
        print("Kernel PCA successfully unrolled the non-linear dataset.")
    else:
        print("Kernel PCA failed to separate the dataset.")
