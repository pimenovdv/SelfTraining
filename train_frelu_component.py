import numpy as np
import json
import os

def frelu(x, b=0):
    """
    Funnel Activation (FReLU)
    f(x) = max(x, b)
    """
    return np.maximum(x, b)

def frelu_derivative(x, b=0):
    """
    Derivative of FReLU.
    f'(x) = 1 if x > b else 0
    """
    return np.where(x > b, 1.0, 0.0)

def frelu_derivative_b(x, b=0):
    """
    Derivative with respect to b.
    f'(b) = 1 if x <= b else 0
    """
    return np.where(x <= b, 1.0, 0.0)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def test_component():
    np.random.seed(42)

    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    true_b = 0.5
    y = np.maximum(X, true_b) + np.random.normal(0, 0.1, X.shape)

    W = np.random.randn(1, 1) * 0.1
    c = np.zeros((1, 1))
    b = np.array([[0.0]])

    learning_rate = 0.01
    epochs = 1000

    initial_loss = 0
    final_loss = 0

    for epoch in range(epochs):
        Z = np.dot(X, W) + c
        A = frelu(Z, b)

        loss = mse_loss(y, A)
        if epoch == 0:
            initial_loss = loss

        dA = mse_loss_derivative(y, A)

        dZ = dA * frelu_derivative(Z, b)
        db = np.sum(dA * frelu_derivative_b(Z, b), axis=0, keepdims=True)

        dW = np.dot(X.T, dZ)
        dc = np.sum(dZ, axis=0, keepdims=True)

        W -= learning_rate * dW
        c -= learning_rate * dc
        b -= learning_rate * db

    final_loss = mse_loss(y, frelu(np.dot(X, W) + c, b))

    print(f"FReLU Component test completed.")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/frelu_results.json", "w") as f:
        json.dump({
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "status": "success",
            "learned_b": float(b[0,0])
        }, f, indent=4)

if __name__ == "__main__":
    test_component()
