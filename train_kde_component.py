import numpy as np

class KDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, X):
        self.data = np.array(X)

    def _gaussian_kernel(self, distance):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (distance / self.bandwidth) ** 2)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.data - x, axis=1)
            kernel_values = self._gaussian_kernel(distances)
            density = np.sum(kernel_values) / (len(self.data) * self.bandwidth)
            predictions.append(density)
        return np.array(predictions)

if __name__ == "__main__":
    print("Testing KDE component...")
    np.random.seed(42)
    # Generate some bimodal data
    data1 = np.random.normal(loc=2.0, scale=0.5, size=50)
    data2 = np.random.normal(loc=7.0, scale=1.0, size=50)
    X_train = np.concatenate([data1, data2]).reshape(-1, 1)

    kde = KDE(bandwidth=0.5)
    kde.fit(X_train)

    # Test point near mode 1, mode 2, and in between
    X_test = np.array([[2.0], [7.0], [4.5]])
    densities = kde.predict(X_test)

    print(f"Density at 2.0 (Mode 1): {densities[0]:.4f}")
    print(f"Density at 7.0 (Mode 2): {densities[1]:.4f}")
    print(f"Density at 4.5 (Valley): {densities[2]:.4f}")

    assert densities[0] > densities[2], "Density at mode 1 should be higher than valley"
    assert densities[1] > densities[2], "Density at mode 2 should be higher than valley"
    print("KDE component verified mathematically.")
