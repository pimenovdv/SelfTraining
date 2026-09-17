import numpy as np

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice Loss.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return 1.0 - (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_loss_derivative(y_true, y_pred, smooth=1e-6):
    """
    Derivative of the Dice Loss with respect to y_pred.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    denominator = np.sum(y_true_f) + np.sum(y_pred_f) + smooth

    term1 = 2.0 * y_true_f * denominator
    term2 = (2.0 * intersection + smooth)

    dD_dy_pred = (term1 - term2) / (denominator ** 2)
    grad = -dD_dy_pred
    return grad.reshape(y_pred.shape)

def test_dice_loss_component():
    print("Testing Dice Loss Component...")

    np.random.seed(42)
    X = np.random.randn(5, 10)
    y_true = np.random.randint(0, 2, size=(5, 10)).astype(float)

    W = np.random.randn(10, 10)
    b = np.random.randn(10)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    lr = 0.5
    epochs = 200

    loss_history = []

    for epoch in range(epochs):
        Z = np.dot(X, W) + b
        y_pred = sigmoid(Z)

        loss = dice_loss(y_true, y_pred)
        loss_history.append(loss)

        dL_dypred = dice_loss_derivative(y_true, y_pred)
        dypred_dZ = y_pred * (1 - y_pred)
        dL_dZ = dL_dypred * dypred_dZ

        dL_dW = np.dot(X.T, dL_dZ)
        dL_db = np.sum(dL_dZ, axis=0)

        W -= lr * dL_dW
        b -= lr * dL_db

        if epoch % 40 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print(f"Final Loss: {loss_history[-1]:.4f}")
    assert loss_history[-1] < loss_history[0], "Loss should decrease over time"
    print("Dice Loss Component test passed successfully.")

if __name__ == "__main__":
    test_dice_loss_component()
