import numpy as np
import json
import os

def bent_identity(x):
    """
    Bent Identity Activation Function.
    f(x) = (np.sqrt(x**2 + 1) - 1) / 2 + x
    """
    return (np.sqrt(x**2 + 1) - 1) / 2 + x

def bent_identity_derivative(x):
    """
    Derivative of Bent Identity.
    f'(x) = x / (2 * np.sqrt(x**2 + 1)) + 1
    """
    return x / (2 * np.sqrt(x**2 + 1)) + 1

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def test_component():
    np.random.seed(42)

    # Generate some simple synthetic data for regression
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    # True mapping: slightly non-linear
    y = bent_identity(X) + np.random.normal(0, 0.1, X.shape)

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
        A = bent_identity(Z)

        loss = mse_loss(y, A)
        if epoch == 0:
            initial_loss = loss

        # Backward pass
        dA = mse_loss_derivative(y, A)
        dZ = dA * bent_identity_derivative(Z)

        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        # Update weights
        W -= learning_rate * dW
        b -= learning_rate * db

    final_loss = mse_loss(y, bent_identity(np.dot(X, W) + b))

    print(f"Bent Identity Component test completed.")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/bent_identity_results.json", "w") as f:
        json.dump({
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "status": "success"
        }, f, indent=4)

if __name__ == "__main__":
    test_component()
