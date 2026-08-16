import numpy as np

def generate_data(n_samples=200):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    # Binary classification problem: XOR-like but simpler
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    # Add some noise
    flip = np.random.rand(n_samples) < 0.1
    y[flip] = -y[flip]
    return X, y

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions

class AdaBoost:
    def __init__(self, n_clf=50):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    error = sum(w[y != predictions])

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)

def main():
    print("Testing AdaBoost Component...")
    X, y = generate_data()
    X_train, y_train = X[:160], y[:160]
    X_test, y_test = X[160:], y[160:]

    clf = AdaBoost(n_clf=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = np.mean(y_pred == y_test)
    print(f"AdaBoost Accuracy: {accuracy * 100:.2f}%")
    assert accuracy > 0.8, "Accuracy is too low, model failed to learn."
    print("AdaBoost implementation is successful.")

if __name__ == "__main__":
    main()
