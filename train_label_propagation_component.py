import numpy as np

class LabelPropagation:
    def __init__(self, max_iter=1000, tol=1e-5, gamma=15.0):
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma

    def fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,) with -1 for unlabeled data
        """
        n_samples = X.shape[0]
        self.classes_ = np.unique(y[y != -1])
        n_classes = len(self.classes_)

        # Initialize label matrix Y
        Y = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            if y[i] != -1:
                idx = np.where(self.classes_ == y[i])[0][0]
                Y[i, idx] = 1.0

        Y_static = np.copy(Y)
        labeled_indices = np.where(y != -1)[0]

        # Compute affinity matrix (RBF kernel)
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    dist_sq = np.sum((X[i] - X[j])**2)
                    W[i, j] = np.exp(-self.gamma * dist_sq)

        # Compute transition matrix T = D^-1 W
        D = np.sum(W, axis=1)
        D[D == 0] = 1.0
        T = W / D[:, np.newaxis]

        # Label propagation
        for _ in range(self.max_iter):
            Y_prev = np.copy(Y)
            Y = T.dot(Y)
            Y[labeled_indices] = Y_static[labeled_indices]

            if np.linalg.norm(Y - Y_prev) < self.tol:
                break

        self.label_distributions_ = Y
        self.transduction_ = self.classes_[np.argmax(Y, axis=1)]
        return self

if __name__ == "__main__":
    print("Testing Label Propagation...")
    np.random.seed(42)
    X1 = np.random.randn(20, 2) * 0.5 + np.array([2, 2])
    X2 = np.random.randn(20, 2) * 0.5 + np.array([-2, -2])
    X = np.vstack((X1, X2))

    true_labels = np.array([0]*20 + [1]*20)

    y = np.copy(true_labels)
    unlabeled_idx = np.concatenate([np.arange(1, 20), np.arange(21, 40)])
    y[unlabeled_idx] = -1

    model = LabelPropagation(gamma=1.0)
    model.fit(X, y)

    accuracy = np.mean(model.transduction_ == true_labels)
    print(f"Semi-supervised Clustering Accuracy: {accuracy * 100:.2f}%")

    if accuracy > 0.95:
        print("Label Propagation mathematical evaluation successful.")
    else:
        print("Label Propagation failed to propagate labels correctly.")
        exit(1)
