import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-6
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

def main():
    print("Initializing Gaussian Naive Bayes experiment...")

    # Generate synthetic dataset
    np.random.seed(42)
    # Class 0: mean=[-2, -2], cov=I
    X0 = np.random.randn(100, 2) + np.array([-2, -2])
    y0 = np.zeros(100)

    # Class 1: mean=[2, 2], cov=I
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    y1 = np.ones(100)

    # Class 2: mean=[-2, 2], cov=I
    X2 = np.random.randn(100, 2) + np.array([-2, 2])
    y2 = np.ones(100) * 2

    X = np.vstack((X0, X1, X2))
    y = np.hstack((y0, y1, y2))

    # Shuffle dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Train-test split
    split = int(0.8 * X.shape[0])
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit model
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    # Predict
    y_pred = gnb.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"Gaussian Naive Bayes training completed.")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
