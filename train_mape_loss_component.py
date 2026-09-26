import numpy as np

def mape_loss(y_true, y_pred, epsilon=1e-8):
    """
    Computes the Mean Absolute Percentage Error (MAPE).

    MAPE = (1/n) * sum(|(y_true - y_pred) / max(epsilon, |y_true|)|)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    denominator = np.maximum(np.abs(y_true), epsilon)

    diff = np.abs((y_true - y_pred) / denominator)
    return np.mean(diff)

def mape_loss_gradient(y_true, y_pred, epsilon=1e-8):
    """
    Computes the gradient of the MAPE loss with respect to y_pred.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = np.maximum(np.abs(y_true), epsilon)

    # Gradient of |x| is sign(x). Here x = (y_true - y_pred) / denominator
    # loss = mean( |y_true - y_pred| / denominator )
    # d(loss)/d(y_pred) = mean( -sign(y_true - y_pred) / denominator )
    # which is the same as sign(y_pred - y_true) / denominator / N

    N = y_true.size
    grad = np.sign(y_pred - y_true) / denominator / N
    return grad

def test_mape_loss():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.8, 3.3, 4.0])

    loss = mape_loss(y_true, y_pred)
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"MAPE Loss: {loss:.4f}")

    grad = mape_loss_gradient(y_true, y_pred)
    print(f"Gradient: {grad}")

    # Numerical gradient check
    eps = 1e-5
    num_grad = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        y_pred_plus = y_pred.copy()
        y_pred_plus[i] += eps
        loss_plus = mape_loss(y_true, y_pred_plus)

        y_pred_minus = y_pred.copy()
        y_pred_minus[i] -= eps
        loss_minus = mape_loss(y_true, y_pred_minus)

        num_grad[i] = (loss_plus - loss_minus) / (2 * eps)

    print(f"Numerical Gradient: {num_grad}")
    assert np.allclose(grad, num_grad, atol=1e-4), "Gradient check failed!"
    print("Gradient check passed!")

if __name__ == "__main__":
    test_mape_loss()
