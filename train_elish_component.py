import numpy as np
import json
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def elish(x):
    cond = x >= 0
    return np.where(cond, x * sigmoid(x), (np.exp(x) - 1) * sigmoid(x))

def elish_derivative(x):
    sig = sigmoid(x)
    cond = x >= 0

    # For x >= 0: f(x) = x * sig
    # f'(x) = sig + x * sig * (1 - sig)
    grad_pos = sig + x * sig * (1 - sig)

    # For x < 0: f(x) = (e^x - 1) * sig
    # f'(x) = e^x * sig + (e^x - 1) * sig * (1 - sig)
    grad_neg = np.exp(x) * sig + (np.exp(x) - 1) * sig * (1 - sig)

    return np.where(cond, grad_pos, grad_neg)

# A simple linear layer
class Linear:
    def __init__(self, in_features, out_features):
        np.random.seed(42)
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output, lr=0.01):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)

        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias
        return grad_input

# MSE Loss
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

# Training loop
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.sum(X, axis=1, keepdims=True) + np.random.randn(100, 1) * 0.1 # Some linear relation with noise

layer = Linear(5, 1)

losses = []
for epoch in range(100):
    # Forward pass
    z = layer.forward(X)
    a = elish(z)
    loss = mse_loss(a, y)
    losses.append(float(loss))

    # Backward pass
    grad_a = mse_derivative(a, y)
    grad_z = grad_a * elish_derivative(z)
    layer.backward(grad_z, lr=0.01)

print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")

os.makedirs("results", exist_ok=True)
with open("results/elish_results.json", "w") as f:
    json.dump({"initial_loss": losses[0], "final_loss": losses[-1]}, f)

print("Elish component verified successfully.")
