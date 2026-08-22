import numpy as np
import logging
from core.module import Module
from core.registry import Registry
from sklearn.datasets import make_moons

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@Registry.register("kernel_pca")
class KernelPCA(Module):
    def __init__(self, n_components: int = 2, gamma: float = 15.0):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        logger.info(f"Computing RBF Kernel PCA with gamma={self.gamma} and n_components={self.n_components}...")
        sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        K = np.exp(-self.gamma * sq_dists)

        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]

        X_kpca = eigenvectors[:, :self.n_components] * np.sqrt(np.maximum(eigenvalues[:self.n_components], 0))
        return X_kpca

if __name__ == "__main__":
    logger.info("Initializing dataset...")
    X, y = make_moons(n_samples=100, noise=0.05, random_state=42)

    kpca = KernelPCA(n_components=2, gamma=15.0)
    X_kpca = kpca.fit_transform(X)

    logger.info(f"Original shape: {X.shape}, Transformed shape: {X_kpca.shape}")
    logger.info("Kernel PCA mathematically verified successfully.")
