import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class FactorizationMachine:
    def __init__(self, num_features, k=4, learning_rate=0.01, epochs=100):
        self.num_features = num_features
        self.k = k
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights
        self.w0 = 0.0
        self.W = np.zeros(num_features)
        self.V = np.random.normal(scale=0.1, size=(num_features, k))

    def predict(self, X):
        """Predict output using FM equation."""
        linear_terms = self.w0 + np.dot(X, self.W)

        # Interactions
        # 1/2 * sum_{f=1}^k ( (sum_{i=1}^n V_{i,f} * X_i)^2 - sum_{i=1}^n (V_{i,f} * X_i)^2 )
        interactions = 0.5 * np.sum(
            np.dot(X, self.V)**2 - np.dot(X**2, self.V**2),
            axis=1
        )

        return sigmoid(linear_terms + interactions)

    def train(self, X, Y):
        """Train FM using SGD."""
        num_samples = X.shape[0]

        for epoch in range(self.epochs):
            total_loss = 0

            for i in range(num_samples):
                x = X[i:i+1]
                y = Y[i]

                # Forward
                pred = self.predict(x)[0]

                # Binary cross-entropy loss
                loss = - (y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
                total_loss += loss

                # Gradients (dL/d_y_hat * d_y_hat/d_out = pred - y)
                grad_out = pred - y

                # Gradient of w0
                self.w0 -= self.learning_rate * grad_out

                # Gradient of W
                self.W -= self.learning_rate * grad_out * x[0]

                # Gradient of V
                for f in range(self.k):
                    v_dot_x = np.dot(x[0], self.V[:, f])
                    for j in range(self.num_features):
                        grad_v = grad_out * (x[0, j] * v_dot_x - self.V[j, f] * x[0, j]**2)
                        self.V[j, f] -= self.learning_rate * grad_v

            if epoch % (self.epochs // 10) == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / num_samples:.4f}")

def main():
    print("--- Training Factorization Machine Component ---")
    np.random.seed(42)

    # Simple synthetic dataset (e.g. user-item interactions with features)
    # 4 features: [User1, User2, Item1, Item2]
    # We want User1 to prefer Item1, User2 to prefer Item2
    X = np.array([
        [1, 0, 1, 0], # U1, I1 -> Like (1)
        [1, 0, 0, 1], # U1, I2 -> Dislike (0)
        [0, 1, 1, 0], # U2, I1 -> Dislike (0)
        [0, 1, 0, 1], # U2, I2 -> Like (1)
    ])

    Y = np.array([1, 0, 0, 1])

    fm = FactorizationMachine(num_features=4, k=2, learning_rate=0.1, epochs=200)
    fm.train(X, Y)

    predictions = fm.predict(X)
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, True: {Y[i]}, Pred: {predictions[i]:.4f}")

    mse = np.mean((predictions - Y)**2)
    print(f"\nMSE: {mse:.4f}")

    if mse < 0.1:
        print("Factorization Machine successfully learned interactions!")
    else:
        print("Factorization Machine failed.")

if __name__ == "__main__":
    main()
