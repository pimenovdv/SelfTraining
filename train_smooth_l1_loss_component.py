import numpy as np

def smooth_l1_loss(y_true, y_pred, beta=1.0):
    diff = y_true - y_pred
    abs_diff = np.abs(diff)

    # Calculate loss
    loss = np.where(abs_diff < beta,
                    0.5 * (diff ** 2) / beta,
                    abs_diff - 0.5 * beta)

    return np.mean(loss)

def smooth_l1_loss_derivative(y_true, y_pred, beta=1.0):
    diff = y_true - y_pred
    abs_diff = np.abs(diff)

    # Gradient w.r.t y_pred
    grad = np.where(abs_diff < beta,
                    -diff / beta,
                    -np.sign(diff))

    return grad / y_true.size

def test_smooth_l1_loss_component():
    print("Testing Smooth L1 Loss component...")

    # Dummy data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y_true = np.random.randn(100, 1)

    # Model parameters
    W = np.random.randn(5, 1) * 0.01
    b = np.zeros((1, 1))

    # Hyperparameters
    learning_rate = 0.1
    epochs = 100
    beta = 1.0

    for epoch in range(epochs):
        # Forward pass
        y_pred = np.dot(X, W) + b

        # Calculate loss
        loss = smooth_l1_loss(y_true, y_pred, beta)

        # Backward pass
        grad_y_pred = smooth_l1_loss_derivative(y_true, y_pred, beta)

        grad_W = np.dot(X.T, grad_y_pred)
        grad_b = np.sum(grad_y_pred, axis=0, keepdims=True)

        # Update parameters
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    print("Smooth L1 Loss component tested successfully.")

if __name__ == "__main__":
    test_smooth_l1_loss_component()
