import numpy as np

def tukey_loss(y_true, y_pred, c=4.685):
    residual = y_pred - y_true
    squared_c = c ** 2
    loss = np.where(
        np.abs(residual) <= c,
        (squared_c / 6.0) * (1.0 - (1.0 - (residual / c) ** 2) ** 3),
        squared_c / 6.0
    )
    return np.mean(loss)

def tukey_gradient(y_true, y_pred, c=4.685):
    residual = y_pred - y_true
    grad = np.where(
        np.abs(residual) <= c,
        residual * (1.0 - (residual / c) ** 2) ** 2,
        0.0
    )
    return grad / y_true.size

def test_component():
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y_true = 3.0 * X + 2.0

    # Introduce some extreme outliers
    y_true[90:] += 50.0

    W = np.random.randn(1, 1)
    b = np.random.randn(1)

    learning_rate = 0.5
    epochs = 500

    print("Training Linear Model with Tukey's Biweight Loss")
    print(f"Initial W: {W[0, 0]:.4f}, b: {b[0]:.4f}")

    for epoch in range(epochs):
        y_pred = np.dot(X, W) + b
        loss = tukey_loss(y_true, y_pred, c=4.685)

        grad_y_pred = tukey_gradient(y_true, y_pred, c=4.685)

        grad_W = np.dot(X.T, grad_y_pred)
        grad_b = np.sum(grad_y_pred)

        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    print(f"Final W: {W[0, 0]:.4f}, b: {b[0]:.4f}")
    print("Optimization finished.")

if __name__ == "__main__":
    test_component()
