import numpy as np

def bce_loss_forward(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def bce_loss_backward(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    grad = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.size
    return grad

def test_component():
    print("Testing Binary Cross Entropy Loss Component...")
    y_true = np.array([[1], [0], [1], [0]])
    y_pred = np.array([[0.9], [0.2], [0.8], [0.1]])
    loss = bce_loss_forward(y_true, y_pred)
    grad = bce_loss_backward(y_true, y_pred)
    print(f"BCE Loss: {loss:.4f}")
    assert loss > 0, "Loss should be positive"
    assert grad.shape == y_pred.shape, "Grad shape mismatch"
    print("Tests passed.")

if __name__ == "__main__":
    test_component()
