import numpy as np

class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1_ratio=0.5, lr=0.1, n_iterations=2000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
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
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) \
                 + self.alpha * self.l1_ratio * np.sign(self.weights) \
                 + self.alpha * (1 - self.l1_ratio) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 2)
    true_weights = np.array([3.5, 0.0])
    true_bias = 1.0
    y = np.dot(X, true_weights) + true_bias + np.random.randn(100) * 0.1

    model = ElasticNetRegression(alpha=0.05, l1_ratio=0.5, lr=0.1, n_iterations=2000)
    model.fit(X, y)
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)

    print("True Weights:", true_weights)
    print("Fitted Weights:", model.weights)
    print("True Bias:", true_bias)
    print("Fitted Bias:", model.bias)
    print("MSE:", mse)
