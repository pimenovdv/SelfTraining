import numpy as np
import os
import json

class Hardshrink:
    def __init__(self, lambd=0.5):
        self.lambd = lambd
        self.x = None

    def forward(self, x):
        self.x = x
        return np.where((x > self.lambd) | (x < -self.lambd), x, 0.0)

    def backward(self, grad_output):
        grad_input = grad_output * np.where((self.x > self.lambd) | (self.x < -self.lambd), 1.0, 0.0)
        return grad_input

def generate_data(num_samples=1000, input_dim=5):
    np.random.seed(42)
    X = np.random.randn(num_samples, input_dim)
    # The target is the sum of hardshrink applied to X
    lambd = 0.5
    Y = np.sum(np.where((X > lambd) | (X < -lambd), X, 0.0), axis=1, keepdims=True)
    return X, Y

class SimpleNN:
    def __init__(self, input_dim, lambd=0.5):
        # Increased initialization to encourage non-zero gradients
        self.W = np.random.randn(input_dim, 1) * 0.5
        self.b = np.zeros((1, 1))
        self.activation = Hardshrink(lambd)

    def forward(self, x):
        self.linear = np.dot(x, self.W) + self.b
        self.out = self.activation.forward(self.linear)
        return self.out

    def backward(self, x, grad_output, learning_rate=0.01):
        grad_linear = self.activation.backward(grad_output)
        grad_W = np.dot(x.T, grad_linear) / x.shape[0]
        grad_b = np.sum(grad_linear, axis=0, keepdims=True) / x.shape[0]

        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b

def train_component():
    X, Y = generate_data()
    model = SimpleNN(input_dim=5, lambd=0.5)

    epochs = 200
    learning_rate = 0.5

    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X)

        # Mean Squared Error Loss
        loss = np.mean((predictions - Y) ** 2)

        # Backward pass
        grad_output = 2 * (predictions - Y)
        model.backward(X, grad_output, learning_rate)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print(f"Final Loss: {loss:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/hardshrink_results.json", "w") as f:
        json.dump({"final_loss": float(loss)}, f)

if __name__ == "__main__":
    train_component()
