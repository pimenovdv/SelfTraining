import numpy as np
import json
import os

def squareplus(x, b=4.0):
    return 0.5 * (x + np.sqrt(x**2 + b))

def squareplus_derivative(x, b=4.0):
    return 0.5 * (1 + x / np.sqrt(x**2 + b))

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

def train_squareplus_component():
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = np.where(X > 0, X, 0) + np.random.normal(0, 0.1, X.shape)

    W1 = np.random.randn(1, 10) * 0.1
    b1 = np.zeros((1, 10))
    W2 = np.random.randn(10, 1) * 0.1
    b2 = np.zeros((1, 1))

    epochs = 1000
    lr = 0.05
    losses = []

    for epoch in range(epochs):
        z1 = X.dot(W1) + b1
        a1 = squareplus(z1, b=0.1)
        z2 = a1.dot(W2) + b2
        y_pred = z2

        loss = mse_loss(y_pred, y)
        losses.append(float(loss))

        dy = mse_loss_derivative(y_pred, y)
        dW2 = a1.T.dot(dy)
        db2 = np.sum(dy, axis=0, keepdims=True)

        da1 = dy.dot(W2.T)
        dz1 = da1 * squareplus_derivative(z1, b=0.1)
        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    print(f"Final Loss: {losses[-1]:.4f}")

    os.makedirs('results', exist_ok=True)
    with open('results/squareplus_results.json', 'w') as f:
        json.dump({'final_loss': losses[-1], 'losses': losses[:10]}, f)

if __name__ == "__main__":
    train_squareplus_component()
