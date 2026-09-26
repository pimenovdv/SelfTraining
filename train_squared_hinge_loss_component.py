import numpy as np

def squared_hinge_loss(y_true, y_pred):
    """
    Computes the Squared Hinge Loss and its gradient.
    Squared Hinge Loss = mean(max(0, 1 - y_true * y_pred)^2)
    y_true should be in {-1, 1}.
    """
    margin = 1 - y_true * y_pred
    loss = np.mean(np.maximum(0, margin) ** 2)

    grad = np.where(margin > 0, -2 * y_true * margin / len(y_true), 0)
    return loss, grad

def test_component():
    y_true = np.array([1, -1, 1, -1])
    y_pred = np.array([0.8, -0.5, -0.2, 1.2]) # margin: 1-0.8=0.2, 1-0.5=0.5, 1-(-0.2)=1.2, 1-(-1.2)=2.2

    loss, grad = squared_hinge_loss(y_true, y_pred)
    print(f"Loss: {loss}")
    print(f"Gradient: {grad}")

if __name__ == "__main__":
    test_component()
