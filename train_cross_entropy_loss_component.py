import numpy as np

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Categorical cross entropy
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

def test_cross_entropy_loss():
    print("Testing Cross Entropy Loss Component...")
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

    loss = cross_entropy_loss(y_true, y_pred)
    print(f"y_true:\n{y_true}")
    print(f"y_pred:\n{y_pred}")
    print(f"Calculated Loss: {loss:.4f}")

    # Compare with sklearn or manual known values
    # For instance, loss = -(log(0.9) + log(0.8) + log(0.6)) / 3
    expected_loss = -(np.log(0.9) + np.log(0.8) + np.log(0.6)) / 3
    print(f"Expected Loss: {expected_loss:.4f}")

    assert np.isclose(loss, expected_loss), f"Loss {loss} != {expected_loss}"
    print("Cross Entropy Loss is correct!")

if __name__ == "__main__":
    test_cross_entropy_loss()
