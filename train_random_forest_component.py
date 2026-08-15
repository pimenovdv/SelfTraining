import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, num_features=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features

    def fit(self, X, y):
        self.num_features = X.shape[1] if not self.num_features else min(self.num_features, X.shape[1])
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples_split and depth <= self.max_depth:
            best_split = self._get_best_split(X, y, num_features)
            if best_split["info_gain"] > 0:
                left = self._build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
                right = self._build_tree(best_split["X_right"], best_split["y_right"], depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left, right)

        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)

    def _get_best_split(self, X, y, num_features):
        best_split = {"info_gain": -1}
        max_info_gain = -float("inf")
        feature_indices = np.random.choice(num_features, self.num_features, replace=False)

        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
                left_dataset, right_dataset = self._split(dataset, feature_index, threshold)
                if len(left_dataset) > 0 and len(right_dataset) > 0:
                    y_curr, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    info_gain = self._information_gain(y_curr, left_y, right_y)
                    if info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "X_left": left_dataset[:, :-1],
                            "y_left": left_y,
                            "X_right": right_dataset[:, :-1],
                            "y_right": right_y,
                            "info_gain": info_gain
                        }
                        max_info_gain = info_gain
        return best_split

    def _split(self, dataset, feature_index, threshold):
        left = np.array([row for row in dataset if row[feature_index] <= threshold])
        right = np.array([row for row in dataset if row[feature_index] > threshold])
        return left, right

    def _information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        return self._gini(parent) - (weight_l * self._gini(l_child) + weight_r * self._gini(r_child))

    def _gini(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini

    def _calculate_leaf_value(self, y):
        Y = list(y)
        return max(Y, key=Y.count)

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

class RandomForest:
    def __init__(self, num_trees=10, min_samples_split=2, max_depth=100, num_features=None):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                num_features=self.num_features
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        num_samples = X.shape[0]
        indices = np.random.choice(num_samples, num_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        predictions = []
        for preds in tree_preds:
            Y = list(preds)
            predictions.append(max(Y, key=Y.count))
        return np.array(predictions)

if __name__ == "__main__":
    print("Testing Random Forest component mathematically...")
    np.random.seed(42)
    # Binary classification synthetic dataset
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X2 = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([1]*50 + [0]*50)

    model = RandomForest(num_trees=5, max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Random Forest accuracy: {accuracy * 100:.2f}%")
    assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"
    print("Random Forest component verified.")
