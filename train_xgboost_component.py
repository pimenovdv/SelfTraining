import numpy as np

class XGBoostTreeRegressor:
    def __init__(self, max_depth=3, min_child_weight=1.0, reg_lambda=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.tree = None

    def fit(self, X, g, h, depth=0):
        if depth >= self.max_depth or np.sum(h) < self.min_child_weight:
            return {'val': -np.sum(g) / (np.sum(h) + self.reg_lambda)}

        best_split = self._find_best_split(X, g, h)
        if best_split is None or best_split['gain'] <= self.gamma:
            return {'val': -np.sum(g) / (np.sum(h) + self.reg_lambda)}

        left_idx = X[:, best_split['feature_idx']] <= best_split['threshold']
        right_idx = ~left_idx

        return {
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'left': self.fit(X[left_idx], g[left_idx], h[left_idx], depth + 1),
            'right': self.fit(X[right_idx], g[right_idx], h[right_idx], depth + 1)
        }

    def _find_best_split(self, X, g, h):
        best_gain = 0.0
        best_split = None
        G = np.sum(g)
        H = np.sum(h)

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold

                GL = np.sum(g[left_idx])
                HL = np.sum(h[left_idx])
                GR = G - GL
                HR = H - HL

                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue

                gain = 0.5 * ((GL**2 / (HL + self.reg_lambda)) +
                              (GR**2 / (HR + self.reg_lambda)) -
                              (G**2 / (H + self.reg_lambda)))

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'gain': gain
                    }
        return best_split

    def predict_row(self, row, node):
        if 'val' in node:
            return node['val']
        if row[node['feature_idx']] <= node['threshold']:
            return self.predict_row(row, node['left'])
        else:
            return self.predict_row(row, node['right'])

    def predict(self, X):
        if self.tree is None:
            return np.zeros(X.shape[0])
        return np.array([self.predict_row(row, self.tree) for row in X])


class XGBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.trees = []
        self.base_pred = 0.0

    def fit(self, X, y):
        self.base_pred = np.mean(y)
        predictions = np.full(y.shape, self.base_pred)

        for _ in range(self.n_estimators):
            # For MSE, gradient is (pred - y), hessian is 1.0
            g = predictions - y
            h = np.ones_like(y)

            tree = XGBoostTreeRegressor(max_depth=self.max_depth, reg_lambda=self.reg_lambda)
            tree.tree = tree.fit(X, g, h)

            update = tree.predict(X)
            predictions += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

if __name__ == "__main__":
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    model = XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, reg_lambda=1.0)
    model.fit(X, y)

    preds = model.predict(X)
    mse = np.mean((preds - y)**2)
    print(f"XGBoost MSE on noisy sine wave: {mse:.4f}")
    assert mse < 0.1, "MSE is too high, model failed to learn."
    print("XGBoost Component verified successfully!")
