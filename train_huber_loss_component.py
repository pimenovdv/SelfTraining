import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Computes the Huber Loss between true and predicted values.

    Huber loss combines MSE and MAE. It is quadratic for small errors
    and linear for large errors, making it robust to outliers.

    Args:
        y_true (np.ndarray): True labels/values.
        y_pred (np.ndarray): Predicted values.
        delta (float): Threshold parameter for transition between quadratic and linear.

    Returns:
        tuple: (loss_value, gradients)
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta

    # Loss computation
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    loss = np.where(is_small_error, squared_loss, linear_loss)
    mean_loss = np.mean(loss)

    # Gradient computation (w.r.t y_pred)
    grad = np.where(is_small_error, -error, -delta * np.sign(error))
    grad /= y_true.size

    return mean_loss, grad

def test_huber_loss():
    np.random.seed(42)
    # Generate true values and some predictions with outliers
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.0, 2.8, 10.0, 4.5]) # 10.0 is an outlier

    delta = 1.0

    print("Testing Huber Loss Component...")
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"Delta: {delta}")

    loss, grad = huber_loss(y_true, y_pred, delta=delta)

    print(f"\nCalculated Huber Loss: {loss:.4f}")
    print(f"Calculated Gradient w.r.t y_pred: \n{grad}")

    # Manual verification for the first element
    expected_error_0 = y_true[0] - y_pred[0] # 1.0 - 1.1 = -0.1
    expected_loss_0 = 0.5 * (-0.1)**2 if abs(expected_error_0) <= delta else delta * (abs(expected_error_0) - 0.5 * delta)
    print(f"\nManual loss for index 0 (y_true=1.0, y_pred=1.1): {expected_loss_0:.4f}")

    # Manual verification for outlier (index 3)
    expected_error_3 = y_true[3] - y_pred[3] # 4.0 - 10.0 = -6.0
    expected_loss_3 = 0.5 * (-6.0)**2 if abs(expected_error_3) <= delta else delta * (abs(expected_error_3) - 0.5 * delta)
    print(f"Manual loss for index 3 (outlier, y_true=4.0, y_pred=10.0): {expected_loss_3:.4f}")

    # Ensure gradients have correct shape
    assert grad.shape == y_pred.shape, "Gradient shape does not match y_pred shape"
    print("\nComponent test completed successfully.")

if __name__ == "__main__":
    test_huber_loss()
