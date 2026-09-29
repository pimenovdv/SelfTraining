import numpy as np

def logcosh_loss(y_true, y_pred):
    """
    Log-Cosh Loss: L(y, y') = sum(log(cosh(y_pred - y_true)))
    """
    diff = y_pred - y_true
    return np.mean(np.log(np.cosh(diff)))

def logcosh_gradient(y_true, y_pred):
    """
    Derivative of Log-Cosh Loss: L'(y, y') = tanh(y_pred - y_true)
    """
    return np.tanh(y_pred - y_true) / y_true.size

def test_logcosh_loss():
    # Simple linear regression with some outliers
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    # y = 2.5 * x + 1.5 + noise
    y_true = 2.5 * X + 1.5 + np.random.randn(100, 1)

    # Introduce some outliers
    y_true[10] += 20.0
    y_true[50] -= 15.0

    # Initialize weights
    W = np.random.randn(1, 1)
    b = np.random.randn(1)

    lr = 0.1
    epochs = 1000

    print("Training Log-Cosh Loss Component...")
    for epoch in range(epochs):
        # Forward pass
        y_pred = np.dot(X, W) + b

        # Compute loss
        loss = logcosh_loss(y_true, y_pred)

        # Backward pass
        dy_pred = logcosh_gradient(y_true, y_pred)
        dW = np.dot(X.T, dy_pred)
        db = np.sum(dy_pred)

        # Update weights
        W -= lr * dW
        b -= lr * db

        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")

    print(f"Final Weights: W = {W[0,0]:.4f}, b = {b[0]:.4f}")

    # True weights: 2.5, 1.5
    # The Log-Cosh loss is robust to outliers, so it should be relatively close to the true weights
    print("Log-Cosh Loss component trained successfully.")

if __name__ == '__main__':
    test_logcosh_loss()
