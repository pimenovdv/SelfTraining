import numpy as np

class ExtremeLearningMachine:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Randomly initialize hidden weights and biases
        self.W_in = np.random.normal(0, 1, (self.input_dim, self.hidden_dim))
        self.b_in = np.random.normal(0, 1, self.hidden_dim)
        self.W_out = None

    def activation(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def train(self, X, y):
        # Calculate hidden layer output matrix H
        H = self.activation(np.dot(X, self.W_in) + self.b_in)

        # Calculate output weights using Moore-Penrose pseudoinverse
        H_pinv = np.linalg.pinv(H)
        self.W_out = np.dot(H_pinv, y)

    def predict(self, X):
        H = self.activation(np.dot(X, self.W_in) + self.b_in)
        return np.dot(H, self.W_out)

def test_elm():
    # Generate some simple synthetic data for regression
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    # y = 2*x0 - x1 + 0.5*x2 + noise
    y_train = 2 * X_train[:, 0] - X_train[:, 1] + 0.5 * X_train[:, 2] + np.random.normal(0, 0.1, 100)

    # Train ELM
    elm = ExtremeLearningMachine(input_dim=5, hidden_dim=20, output_dim=1)
    elm.train(X_train, y_train)

    # Test data
    X_test = np.random.rand(20, 5)
    y_test = 2 * X_test[:, 0] - X_test[:, 1] + 0.5 * X_test[:, 2]

    predictions = elm.predict(X_test)

    mse = np.mean((predictions - y_test) ** 2)
    print("Extreme Learning Machine (ELM) Test")
    print(f"MSE on Test Data: {mse:.4f}")
    if mse < 0.1:
        print("ELM learned the regression mapping successfully.")
    else:
        print("ELM failed to learn adequately.")

if __name__ == "__main__":
    test_elm()
