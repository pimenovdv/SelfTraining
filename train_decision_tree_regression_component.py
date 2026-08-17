import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples_split and depth <= self.max_depth:
            best_split = self._get_best_split(X, y, num_features)
            if best_split["var_reduction"] > 0:
                left = self._build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
                right = self._build_tree(best_split["X_right"], best_split["y_right"], depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left, right)

        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)

    def _get_best_split(self, X, y, num_features):
        best_split = {"var_reduction": -1}
        max_var_reduction = -float("inf")

        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
                left_dataset, right_dataset = self._split(dataset, feature_index, threshold)
                if len(left_dataset) > 0 and len(right_dataset) > 0:
                    y_parent, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    var_reduction = self._variance_reduction(y_parent, left_y, right_y)
                    if var_reduction > max_var_reduction:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "X_left": left_dataset[:, :-1],
                            "y_left": left_y,
                            "X_right": right_dataset[:, :-1],
                            "y_right": right_y,
                            "var_reduction": var_reduction
                        }
                        max_var_reduction = var_reduction
        return best_split

    def _split(self, dataset, feature_index, threshold):
        left = np.array([row for row in dataset if row[feature_index] <= threshold])
        right = np.array([row for row in dataset if row[feature_index] > threshold])
        return left, right

    def _variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        return np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))

    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def predict(self, X):
        return np.array([self._make_prediction(x, self.root) for x in X])

    def _make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)

if __name__ == "__main__":
    print("Testing Decision Tree Regression component mathematically...")
    np.random.seed(42)
    # Non-linear regression synthetic dataset (sine wave)
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(16)) # Add noise

    model = DecisionTreeRegressor(min_samples_split=2, max_depth=5)
    model.fit(X, y)
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    print(f"Decision Tree Regression MSE: {mse:.4f}")
    assert mse < 0.2, f"Expected MSE < 0.2, got {mse}"
    print("Decision Tree Regression component verified.")
