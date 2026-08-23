import numpy as np

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.height = 0

    def fit(self, X, current_height=0):
        self.size = len(X)
        self.height = current_height

        if current_height >= self.height_limit or self.size <= 1:
            return self

        num_features = X.shape[1]
        self.split_feature = np.random.randint(0, num_features)
        min_val = X[:, self.split_feature].min()
        max_val = X[:, self.split_feature].max()

        if min_val == max_val:
            return self

        self.split_value = np.random.uniform(min_val, max_val)

        left_mask = X[:, self.split_feature] < self.split_value
        right_mask = ~left_mask

        self.left = IsolationTree(self.height_limit).fit(X[left_mask], current_height + 1)
        self.right = IsolationTree(self.height_limit).fit(X[right_mask], current_height + 1)

        return self

    def path_length(self, x):
        if self.left is None and self.right is None:
            if self.size <= 1:
                return self.height
            return self.height + 2.0 * (np.log(self.size - 1) + 0.5772156649) - (2.0 * (self.size - 1) / self.size)

        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x)
        else:
            return self.right.path_length(x)

class IsolationForest:
    def __init__(self, num_trees=100, sample_size=256):
        self.num_trees = num_trees
        self.sample_size = sample_size
        self.trees = []
        self.c = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.sample_size = min(self.sample_size, n_samples)
        height_limit = int(np.ceil(np.log2(self.sample_size)))

        if self.sample_size > 1:
            self.c = 2.0 * (np.log(self.sample_size - 1) + 0.5772156649) - (2.0 * (self.sample_size - 1) / self.sample_size)
        else:
            self.c = 1.0

        for _ in range(self.num_trees):
            indices = np.random.choice(n_samples, self.sample_size, replace=False)
            X_sample = X[indices]
            tree = IsolationTree(height_limit).fit(X_sample)
            self.trees.append(tree)

    def anomaly_score(self, X):
        scores = []
        for x in X:
            path_lengths = [tree.path_length(x) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            score = 2.0 ** (-avg_path_length / self.c)
            scores.append(score)
        return np.array(scores)

if __name__ == "__main__":
    np.random.seed(42)
    # Normal data
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(200, 2))
    # Anomalies
    X_anomaly = np.random.uniform(low=-10.0, high=10.0, size=(20, 2))

    X = np.vstack([X_normal, X_anomaly])

    print("Training Isolation Forest mathematically...")
    forest = IsolationForest(num_trees=100, sample_size=64)
    forest.fit(X)

    scores = forest.anomaly_score(X)

    # Check if anomalies have higher scores (closer to 1) than normal data
    normal_scores = scores[:200]
    anomaly_scores = scores[200:]

    mean_normal_score = np.mean(normal_scores)
    mean_anomaly_score = np.mean(anomaly_scores)

    print(f"Mean Normal Score: {mean_normal_score:.4f}")
    print(f"Mean Anomaly Score: {mean_anomaly_score:.4f}")

    if mean_anomaly_score > mean_normal_score:
        print("Success: Anomalies successfully isolated.")
    else:
        print("Failure: Anomalies not detected correctly.")
