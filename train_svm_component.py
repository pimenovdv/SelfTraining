"""
Support Vector Machine (SVM) Component

This script implements a Linear Support Vector Machine using subgradient descent
on the hinge loss objective function. It demonstrates maximum margin classification.
"""

import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # y must be -1 or 1
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

def main():
    print("Testing Linear SVM Component...")
    np.random.seed(42)

    # Create simple linearly separable data
    X = np.concatenate([np.random.randn(50, 2) - 2, np.random.randn(50, 2) + 2])
    y = np.concatenate([-np.ones(50), np.ones(50)])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    svm = LinearSVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    svm.fit(X, y)

    predictions = svm.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy:.4f}")

    assert accuracy > 0.9, "SVM should easily separate this linearly separable dataset"
    print("SVM test passed!")

if __name__ == "__main__":
    main()
