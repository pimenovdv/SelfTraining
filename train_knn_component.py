import numpy as np
from collections import Counter

def generate_data(num_samples=200):
    np.random.seed(42)
    X0 = np.random.randn(num_samples // 2, 2) + np.array([2, 2])
    y0 = np.zeros(num_samples // 2)
    X1 = np.random.randn(num_samples // 2, 2) + np.array([-2, -2])
    y1 = np.ones(num_samples // 2)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

if __name__ == "__main__":
    print("Initializing KNN component testing...")
    X, y = generate_data(200)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    knn = KNN(k=5)
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)
    accuracy = np.mean(predictions == y_test)

    print(f"KNN Accuracy: {accuracy * 100:.2f}%")
    if accuracy > 0.9:
        print("KNN test passed.")
    else:
        print("KNN test failed.")
