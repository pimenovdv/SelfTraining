import numpy as np

class OneClassSVM:
    def __init__(self, kernel='rbf', nu=0.1, gamma=0.1, learning_rate=0.001, epochs=1000):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = None
        self.X_train = None
        self.rho = None

    def _rbf_kernel(self, X1, X2):
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * sq_dists)

    def _compute_kernel(self, X1, X2):
        if self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Kernel '{self.kernel}' not supported.")

    def fit(self, X):
        n_samples = X.shape[0]
        self.X_train = X
        # Initialize alpha evenly
        self.alpha = np.ones(n_samples) / n_samples

        K = self._compute_kernel(X, X)
        C = 1.0 / (self.nu * n_samples)

        # We solve the dual problem using simple gradient descent with projection.
        # min 0.5 * alpha^T K alpha
        # s.t. 0 <= alpha_i <= C
        #      sum(alpha_i) = 1

        for _ in range(self.epochs):
            # Gradient is K * alpha
            grad = np.dot(K, self.alpha)

            # Step in negative gradient direction
            self.alpha -= self.learning_rate * grad

            # Projection onto the intersection of the box [0, C] and the hyperplane sum(a)=1
            # We can use an iterative projection (Dykstra's algorithm or simply alternating)
            for _ in range(10):
                 # Project onto hyperplane sum(a) = 1
                 self.alpha -= (np.sum(self.alpha) - 1.0) / n_samples
                 # Project onto box
                 self.alpha = np.clip(self.alpha, 0, C)

        # To find rho (the threshold), we can use the support vectors
        # A support vector is one where 0 < alpha_i < C
        eps = 1e-6
        sv_indices = np.where((self.alpha > eps) & (self.alpha < C - eps))[0]

        if len(sv_indices) > 0:
            rho = np.mean(np.dot(K[sv_indices, :], self.alpha))
        else:
            # If no strict support vectors, we use the ones that are at the upper bound
            sv_indices = np.where(self.alpha > eps)[0]
            if len(sv_indices) > 0:
                 rho = np.min(np.dot(K[sv_indices, :], self.alpha))
            else:
                 rho = 0.0

        # Alternatively, for OCSVM, nu is roughly the proportion of outliers.
        # We can just sort the scores and pick the threshold such that nu fraction are negative.
        # This is a robust way to set the threshold if the optimization wasn't perfect.
        scores = np.dot(K, self.alpha)
        # We want nu fraction to have score < rho, so rho is the nu-th quantile.
        self.rho = np.quantile(scores, self.nu)

        return self

    def decision_function(self, X):
        K = self._compute_kernel(X, self.X_train)
        return np.dot(K, self.alpha) - self.rho

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


if __name__ == '__main__':
    np.random.seed(42)
    # Generate normal data (spherical)
    X_normal = np.random.randn(200, 2) * 0.5

    # Generate some outliers (far away)
    X_outliers = np.random.uniform(low=-5, high=5, size=(50, 2))
    # Remove outliers that accidentally fell into the normal distribution
    X_outliers = X_outliers[np.linalg.norm(X_outliers, axis=1) > 2.0]

    # Train One-Class SVM on normal data
    # nu = 0.05 implies we expect ~5% outliers in the training set
    oc_svm = OneClassSVM(kernel='rbf', nu=0.05, gamma=0.5, epochs=1000, learning_rate=0.01)
    oc_svm.fit(X_normal)

    # Predict on normal data and outliers
    preds_normal = oc_svm.predict(X_normal)
    preds_outliers = oc_svm.predict(X_outliers)

    # Evaluate
    normal_accuracy = np.mean(preds_normal == 1)
    outlier_accuracy = np.mean(preds_outliers == -1)

    print(f"Normal Data Accuracy (True Positives): {normal_accuracy * 100:.2f}%")
    print(f"Outlier Detection Accuracy (True Negatives): {outlier_accuracy * 100:.2f}%")

    assert normal_accuracy > 0.8, "Model failed to capture normal data properly."
    assert outlier_accuracy > 0.8, "Model failed to detect outliers."
    print("One-Class SVM component implemented and successfully verified.")
