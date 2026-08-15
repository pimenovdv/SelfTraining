import numpy as np

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)

            # Scatter within
            SW += np.dot((X_c - mean_c).T, (X_c - mean_c))

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            # Scatter between
            SB += n_c * np.dot(mean_diff, mean_diff.T)

        # A = SW^-1 * SB
        A = np.linalg.pinv(SW).dot(SB)

        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)

if __name__ == "__main__":
    np.random.seed(42)
    mean1 = [2, 2, 2]
    cov1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    X1 = np.random.multivariate_normal(mean1, cov1, 50)
    y1 = np.zeros(50)

    mean2 = [8, 8, 8]
    cov2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    X2 = np.random.multivariate_normal(mean2, cov2, 50)
    y2 = np.ones(50)

    mean3 = [8, 2, 2]
    cov3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    X3 = np.random.multivariate_normal(mean3, cov3, 50)
    y3 = np.full(50, 2)

    X = np.vstack((X1, X2, X3))
    y = np.hstack((y1, y2, y3))

    print("Data shape:", X.shape)
    lda = LDA(n_components=2)
    lda.fit(X, y)
    X_projected = lda.transform(X)
    print("Projected shape:", X_projected.shape)
    print("Optimization finished successfully.")
