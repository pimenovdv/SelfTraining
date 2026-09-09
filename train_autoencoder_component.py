import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

class Autoencoder:
    def __init__(self, input_dim, hidden_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b2 = np.zeros((1, input_dim))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2
        return self.a2

    def backward(self, X, y_true, y_pred, learning_rate):
        d_loss = mse_derivative(y_true, y_pred)
        d_z2 = d_loss

        dW2 = np.dot(self.a1.T, d_z2)
        db2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * relu_derivative(self.z1)

        dW1 = np.dot(X.T, d_z1)
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

def main():
    np.random.seed(42)
    # Synthetic dataset
    X = np.random.rand(1000, 10)

    model = Autoencoder(input_dim=10, hidden_dim=5)
    learning_rate = 0.1
    epochs = 2000

    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = mse_loss(X, y_pred)
        model.backward(X, X, y_pred, learning_rate)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    final_pred = model.forward(X)
    final_loss = mse_loss(X, final_pred)
    print(f"Final Loss: {final_loss:.4f}")
    if final_loss < 0.05:
        print("Successfully trained Autoencoder component.")
    else:
        print("Failed to minimize loss sufficiently.")

if __name__ == '__main__':
    main()
