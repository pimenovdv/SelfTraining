import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, lr=0.1, n_iterations=2000):
        self.alpha = alpha
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Subgradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + self.alpha * np.sign(self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    np.random.seed(42)
    # Generate some synthetic data
    X = np.random.rand(100, 2)
    # True weights: feature 1 is useful, feature 2 is not
    true_weights = np.array([3.5, 0.0])
    true_bias = 1.0
    y = np.dot(X, true_weights) + true_bias + np.random.randn(100) * 0.1

    # Train Lasso Regression
    model = LassoRegression(alpha=0.05, lr=0.1, n_iterations=2000)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Evaluate
    mse = np.mean((predictions - y) ** 2)

    print("True Weights:", true_weights)
    print("Fitted Weights:", model.weights)
    print("True Bias:", true_bias)
    print("Fitted Bias:", model.bias)
    print("MSE:", mse)
