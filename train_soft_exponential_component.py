import numpy as np
import json
import os

def soft_exponential(x, alpha):
    """
    Soft Exponential Activation Function.
    If alpha < 0: -ln(1 - alpha * (x + alpha)) / alpha
    If alpha = 0: x
    If alpha > 0: (exp(alpha * x) - 1) / alpha + alpha
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    if alpha < 0:
        result = -np.log(1 - alpha * (x + alpha)) / alpha
    elif alpha == 0:
        result = x
    else:
        result = (np.exp(alpha * x) - 1) / alpha + alpha

    return result

def soft_exponential_derivative(x, alpha):
    """
    Derivative of Soft Exponential w.r.t x.
    If alpha < 0: 1 / (1 - alpha * (x + alpha))
    If alpha = 0: 1
    If alpha > 0: exp(alpha * x)
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    if alpha < 0:
        result = 1.0 / (1 - alpha * (x + alpha))
    elif alpha == 0:
        result = np.ones_like(x)
    else:
        result = np.exp(alpha * x)

    return result

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def test_component():
    np.random.seed(42)

    # Generate some simple synthetic data for regression
    X = np.linspace(-3, 3, 100).reshape(-1, 1)

    # We will test alpha = 0.5
    alpha_test = 0.5
    # True mapping
    y = soft_exponential(X, alpha_test) + np.random.normal(0, 0.1, X.shape)

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
        A = soft_exponential(Z, alpha_test)

        loss = mse_loss(y, A)
        if epoch == 0:
            initial_loss = loss

        # Backward pass
        dA = mse_loss_derivative(y, A)
        dZ = dA * soft_exponential_derivative(Z, alpha_test)

        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        # Update weights
        W -= learning_rate * dW
        b -= learning_rate * db

    final_loss = mse_loss(y, soft_exponential(np.dot(X, W) + b, alpha_test))

    print(f"Soft Exponential Component test completed.")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/soft_exponential_results.json", "w") as f:
        json.dump({
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "status": "success"
        }, f, indent=4)

if __name__ == "__main__":
    test_component()
