import numpy as np

def softplus(x):
    return np.logaddexp(0, x)

def softplus_derivative(x):
    return 1 / (1 + np.exp(-x))

class SoftplusNetworkClassification:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 1.0
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 1.0
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = softplus(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2)) # Sigmoid
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        # BCE loss derivative for sigmoid output is (A2 - y)
        dZ2 = (self.A2 - y) / m
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * softplus_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000):
        for i in range(epochs):
            pred = self.forward(X)
            self.backward(X, y)

if __name__ == "__main__":
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    net = SoftplusNetworkClassification(input_size=2, hidden_size=8, output_size=1, learning_rate=1.0)
    net.train(X, y, epochs=5000)
    preds = net.forward(X)
    print("Predictions:")
    print(preds)

    loss = -np.mean(y * np.log(preds + 1e-15) + (1 - y) * np.log(1 - preds + 1e-15))
    print(f"Final BCE Loss: {loss:.4f}")

    assert np.all(preds[1:3] > 0.5) and np.all(preds[[0,3]] < 0.5), "Failed to learn XOR"
    print("Successfully learned XOR with Softplus activation.")
