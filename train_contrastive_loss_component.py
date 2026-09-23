import numpy as np

def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Calculates the contrastive loss.

    y_true: binary label, 1 for similar (positive pair), 0 for dissimilar (negative pair).
    y_pred: predicted distance between the pair of inputs (e.g., Euclidean distance).
    margin: margin for dissimilar pairs.
    """
    loss = y_true * np.square(y_pred) + (1 - y_true) * np.square(np.maximum(0, margin - y_pred))
    return np.mean(loss)

def contrastive_loss_gradient(y_true, y_pred, margin=1.0):
    """
    Calculates the gradient of the contrastive loss with respect to the predicted distance.
    """
    grad = 2 * y_true * y_pred - 2 * (1 - y_true) * np.maximum(0, margin - y_pred)
    return grad / y_true.size

def test_contrastive_loss():
    print("Testing Contrastive Loss Component...")

    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([0.1, 0.8, 0.2, 1.5, 0.0, 0.9])

    margin = 1.0

    loss = contrastive_loss(y_true, y_pred, margin)
    grad = contrastive_loss_gradient(y_true, y_pred, margin)

    print(f"y_true: {y_true}")
    print(f"y_pred distances: {y_pred}")
    print(f"Calculated Loss: {loss:.4f}")
    print(f"Calculated Gradients: {grad}")

    expected_loss = []
    expected_grad = []
    for i in range(len(y_true)):
        yt = y_true[i]
        yp = y_pred[i]

        l = yt * yp**2 + (1 - yt) * max(0, margin - yp)**2
        expected_loss.append(l)

        if yt == 1:
            g = 2 * yp
        else:
            if margin > yp:
                g = -2 * (margin - yp)
            else:
                g = 0
        expected_grad.append(g)

    expected_loss = np.mean(expected_loss)
    expected_grad = np.array(expected_grad) / len(y_true)

    print(f"Expected Loss: {expected_loss:.4f}")
    print(f"Expected Gradients: {expected_grad}")

    assert np.isclose(loss, expected_loss), "Loss calculation is incorrect."
    assert np.allclose(grad, expected_grad), "Gradient calculation is incorrect."

    print("Contrastive Loss component successfully tested and verified.")

if __name__ == "__main__":
    test_contrastive_loss()
