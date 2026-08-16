import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.value_left = None
        self.value_right = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_mse = float('inf')

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                val_left = np.mean(y[left_mask])
                val_right = np.mean(y[right_mask])

                pred = np.zeros(n_samples)
                pred[left_mask] = val_left
                pred[right_mask] = val_right

                mse = np.mean((y - pred) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    self.feature_idx = feature_idx
                    self.threshold = threshold
                    self.value_left = val_left
                    self.value_right = val_right

    def predict(self, X):
        n_samples = X.shape[0]
        pred = np.zeros(n_samples)
        if self.feature_idx is None:
            return pred

        left_mask = X[:, self.feature_idx] <= self.threshold
        pred[left_mask] = self.value_left
        pred[~left_mask] = self.value_right
        return pred

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        pred = np.full(len(y), self.initial_pred)

        for i in range(self.n_estimators):
            residuals = y - pred
            stump = DecisionStump()
            stump.fit(X, residuals)
            self.estimators.append(stump)
            pred += self.learning_rate * stump.predict(X)

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_pred)
        for stump in self.estimators:
            pred += self.learning_rate * stump.predict(X)
        return pred

if __name__ == "__main__":
    np.random.seed(42)
    # Generate noisy sine wave
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel()
    y += 0.1 * np.random.randn(100)

    gbm = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
    gbm.fit(X, y)
    preds = gbm.predict(X)

    mse = np.mean((y - preds) ** 2)
    print(f"Gradient Boosting Regressor MSE: {mse:.4f}")
    assert mse < 0.1, "MSE is too high, model failed to learn."
    print("Gradient Boosting Regressor successfully fitted the data!")
