import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

if __name__ == "__main__":
    print("Testing Logistic Regression component mathematically...")
    np.random.seed(42)
    # Binary classification synthetic dataset
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X2 = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([1]*50 + [0]*50)

    model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Logistic Regression accuracy: {accuracy * 100:.2f}%")
    assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"
    print("Logistic Regression component verified.")
