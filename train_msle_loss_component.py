import numpy as np
import json
import os

def msle_loss(y_true, y_pred):
    """
    Computes the Mean Squared Logarithmic Error (MSLE) Loss and its gradient.
    Assumes y_true >= 0 and y_pred >= 0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    loss = np.mean((log_pred - log_true) ** 2)

    N = y_true.size
    grad = (2.0 / N) * ((log_pred - log_true) / (1.0 + y_pred))

    return loss, grad

def test_msle_loss():
    np.random.seed(42)

    X = np.random.rand(100, 5)
    y_true = np.random.rand(100, 1) * 10

    W = np.random.randn(5, 1) * 0.1
    b = np.zeros((1, 1))

    learning_rate = 0.5
    losses = []

    for epoch in range(200):
        linear_out = np.dot(X, W) + b
        y_pred = np.maximum(linear_out, 0)

        loss, grad = msle_loss(y_true, y_pred)
        losses.append(float(loss))

        d_linear = grad.copy()
        d_linear[linear_out < 0] = 0

        dW = np.dot(X.T, d_linear)
        db = np.sum(d_linear, axis=0, keepdims=True)

        W -= learning_rate * dW
        b -= learning_rate * db

    print(f"Final MSLE Loss: {losses[-1]:.4f}")

    os.makedirs('results', exist_ok=True)
    with open('results/msle_loss_results.json', 'w') as f:
        json.dump({'losses': losses}, f)

if __name__ == '__main__':
    test_msle_loss()
