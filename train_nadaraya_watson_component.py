import numpy as np

class NadarayaWatsonRegression:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.X_train = None
        self.y_train = None

    def _gaussian_kernel(self, distances):
        return (1.0 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distances / self.bandwidth) ** 2)

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            distances = np.linalg.norm(self.X_train - x, axis=1)
            weights = self._gaussian_kernel(distances)
            sum_weights = np.sum(weights)
            if sum_weights == 0:
                y_pred[i] = np.mean(self.y_train)
            else:
                y_pred[i] = np.sum(weights * self.y_train) / sum_weights
        return y_pred

def test_nadaraya_watson():
    print("Testing Nadaraya-Watson Regression Component...")
    np.random.seed(42)
    X_train = np.linspace(-3, 3, 50).reshape(-1, 1)
    y_train = np.sin(X_train).flatten() + np.random.normal(0, 0.1, 50)

    model = NadarayaWatsonRegression(bandwidth=0.5)
    model.fit(X_train, y_train)

    X_test = np.array([[-1.5], [0.0], [1.5]])
    y_pred = model.predict(X_test)

    print(f"Predicted values: {y_pred}")
    print(f"Expected approx: {np.sin(X_test).flatten()}")

    mse = np.mean((y_pred - np.sin(X_test).flatten())**2)
    print(f"MSE on test points: {mse:.4f}")

    assert mse < 0.05, "MSE is too high, model failed to fit the data."
    print("Nadaraya-Watson Regression testing successful!")

if __name__ == "__main__":
    test_nadaraya_watson()
