import numpy as np

class HuberRegressor:
    def __init__(self, delta=1.0, learning_rate=0.01, epochs=1000):
        self.delta = delta
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            is_small_error = np.abs(error) <= self.delta

            grad_error = np.where(is_small_error, error, self.delta * np.sign(error))

            dw = (1 / n_samples) * np.dot(X.T, grad_error)
            db = (1 / n_samples) * np.sum(grad_error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    # Generate synthetic data with outliers
    np.random.seed(42)
    X = np.random.randn(100, 1) * 2
    y = 3 * X.squeeze() + 5 + np.random.randn(100) * 0.5

    # Add outliers
    y[::10] += 20 * np.random.randn(10)

    model = HuberRegressor(delta=1.5, learning_rate=0.05, epochs=1000)
    model.fit(X, y)

    y_pred = model.predict(X)

    # We evaluate without outliers for the "true" MSE
    mask_inliers = np.ones(100, dtype=bool)
    mask_inliers[::10] = False

    true_mse = np.mean((y[mask_inliers] - y_pred[mask_inliers])**2)
    print(f"Huber Regression completed.")
    print(f"Learned weights: {model.weights[0]:.4f}, Learned bias: {model.bias:.4f}")
    print(f"True MSE (on inliers): {true_mse:.4f}")
