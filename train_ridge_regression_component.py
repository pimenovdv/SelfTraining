import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Add a column of ones to X for the bias term
        X_b = np.c_[np.ones((n_samples, 1)), X]

        # Create the identity matrix for regularization
        # We don't regularize the bias term, so the first element is 0
        I = np.eye(n_features + 1)
        I[0, 0] = 0

        # Calculate the weights using the closed-form solution:
        # w = (X^T X + alpha * I)^-1 X^T y
        A = X_b.T.dot(X_b) + self.alpha * I
        b = X_b.T.dot(y)

        theta = np.linalg.inv(A).dot(b)

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return X.dot(self.weights) + self.bias

if __name__ == "__main__":
    np.random.seed(42)
    # Generate some synthetic data
    X = np.random.rand(100, 2)
    true_weights = np.array([3.5, -2.0])
    true_bias = 1.0
    y = X.dot(true_weights) + true_bias + np.random.randn(100) * 0.1

    # Train Ridge Regression
    model = RidgeRegression(alpha=0.1)
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
