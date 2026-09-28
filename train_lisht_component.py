import numpy as np
import os
import json

def lisht(x):
    return x * np.tanh(x)

def lisht_derivative(x):
    tanh_x = np.tanh(x)
    return tanh_x + x * (1.0 - tanh_x**2)

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = lisht(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        # Use sigmoid for output
        self.a2 = 1.0 / (1.0 + np.exp(-self.z2))
        return self.a2

    def backward(self, x, y):
        m = x.shape[0]

        # Loss derivative for binary cross entropy with sigmoid
        dz2 = self.a2 - y

        dw2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * lisht_derivative(self.z1)

        dw1 = (1 / m) * np.dot(x.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1

def train_component():
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    model = SimpleNN(2, 4, 1, learning_rate=0.5)

    epochs = 5000
    for epoch in range(epochs):
        predictions = model.forward(X)
        model.backward(X, Y)

        if epoch % 1000 == 0:
            loss = -np.mean(Y * np.log(predictions + 1e-8) + (1 - Y) * np.log(1 - predictions + 1e-8))
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    final_predictions = model.forward(X)
    print("Final predictions:")
    print(final_predictions)

    results = {
        "loss": float(-np.mean(Y * np.log(final_predictions + 1e-8) + (1 - Y) * np.log(1 - final_predictions + 1e-8))),
        "predictions": final_predictions.tolist()
    }

    os.makedirs("results", exist_ok=True)
    with open("results/lisht_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    train_component()
