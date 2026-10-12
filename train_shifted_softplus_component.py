import numpy as np
import json
import os

def shifted_softplus(x):
    """
    Shifted Softplus Activation Function.
    f(x) = ln(1 + e^x) - ln(2)
    """
    return np.log1p(np.exp(np.clip(x, -500, 500))) - np.log(2)

def shifted_softplus_derivative(x):
    """
    Derivative of Shifted Softplus.
    f'(x) = 1 / (1 + e^-x) (which is the sigmoid function)
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def test_component():
    np.random.seed(42)

    # Generate some simple synthetic data for regression
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    # True mapping: non-linear based on Shifted Softplus
    y = shifted_softplus(X) + np.random.normal(0, 0.1, X.shape)

    # Model parameters (single linear layer + activation)
    W = np.random.randn(1, 1) * 0.1
    b = np.zeros((1, 1))

    learning_rate = 0.01
    epochs = 1000

    initial_loss = 0
    final_loss = 0

    for epoch in range(epochs):
        # Forward pass
        Z = np.dot(X, W) + b
        A = shifted_softplus(Z)

        loss = mse_loss(y, A)
        if epoch == 0:
            initial_loss = loss

        # Backward pass
        dA = mse_loss_derivative(y, A)
        dZ = dA * shifted_softplus_derivative(Z)

        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        # Update weights
        W -= learning_rate * dW
        b -= learning_rate * db

    final_loss = mse_loss(y, shifted_softplus(np.dot(X, W) + b))

    print(f"Shifted Softplus Component test completed.")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/shifted_softplus_results.json", "w") as f:
        json.dump({
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "status": "success"
        }, f, indent=4)

if __name__ == "__main__":
    test_component()
