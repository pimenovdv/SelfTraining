import numpy as np

def hellinger_distance_loss(y_pred, y_true):
    """
    Computes the Hellinger distance loss and its gradient with respect to y_pred.
    y_pred and y_true must be valid probability distributions (sum to 1 along the class axis).
    """
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1.0)

    # The loss here is 1/2 * sum((sqrt(p) - sqrt(q))^2), which is the squared Hellinger distance.
    loss = np.mean(0.5 * np.sum((np.sqrt(y_pred) - np.sqrt(y_true))**2, axis=1))

    # Gradient w.r.t y_pred. Scaled by batch size (y_pred.shape[0]) for mean loss.
    grad = (np.sqrt(y_pred) - np.sqrt(y_true)) / (2.0 * np.sqrt(y_pred) * y_pred.shape[0])

    return loss, grad

def test_hellinger_distance_loss():
    np.random.seed(42)
    # Target distribution
    y_true = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    # Initialize parameters for a simple linear model mapping inputs to probabilities
    X = np.array([[0.5, -0.2], [-0.5, 0.8]])
    W = np.random.randn(2, 3)
    b = np.zeros((1, 3))

    learning_rate = 0.5

    print("Initial W:\n", W)
    for epoch in range(50):
        # Forward pass
        logits = np.dot(X, W) + b
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Loss and gradient
        loss, d_y_pred = hellinger_distance_loss(y_pred, y_true)

        # Backward pass through softmax
        d_logits = np.zeros_like(logits)
        for i in range(y_pred.shape[0]):
            p = y_pred[i].reshape(-1, 1)
            jacobian = np.diagflat(p) - np.dot(p, p.T)
            d_logits[i] = np.dot(jacobian, d_y_pred[i])

        # Gradients for W and b
        dW = np.dot(X.T, d_logits)
        db = np.sum(d_logits, axis=0, keepdims=True)

        # Update parameters
        W -= learning_rate * dW
        b -= learning_rate * db

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    print("Final y_pred:\n", y_pred)
    assert loss < 0.1, "Loss did not decrease sufficiently."
    print("Component training successful.")

if __name__ == "__main__":
    test_hellinger_distance_loss()
