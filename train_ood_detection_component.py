import numpy as np

class OODDetectionMahalanobis:
    def __init__(self):
        self.mean = None
        self.cov_inv = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(X.shape[1]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)

    def score(self, X):
        diff = X - self.mean
        left = np.dot(diff, self.cov_inv)
        dist = np.sum(left * diff, axis=1)
        return dist

def main():
    print("Initializing Out-of-Distribution Detection (Mahalanobis) component...")

    np.random.seed(42)
    X_train = np.random.randn(1000, 5)

    detector = OODDetectionMahalanobis()
    print("Fitting Mahalanobis detector on in-distribution data...")
    detector.fit(X_train)

    X_test_in = np.random.randn(100, 5)
    scores_in = detector.score(X_test_in)

    X_test_out = np.random.randn(100, 5) * 3 + 5
    scores_out = detector.score(X_test_out)

    mean_in = np.mean(scores_in)
    mean_out = np.mean(scores_out)

    print(f"Average In-Distribution Score (Squared Mahalanobis Distance): {mean_in:.4f}")
    print(f"Average Out-of-Distribution Score: {mean_out:.4f}")

    if mean_out > mean_in * 2:
        print("Out-of-Distribution samples successfully detected. Generalization analysis passed.")
    else:
        print("Failed to detect Out-of-Distribution samples.")

if __name__ == "__main__":
    main()
