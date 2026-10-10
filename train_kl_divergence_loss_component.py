import numpy as np

def kl_divergence_forward(y_true, y_pred, epsilon=1e-15):
    """
    Computes the KL Divergence Loss.
    y_true: true probabilities (P)
    y_pred: predicted probabilities (Q)
    """
    y_true = np.clip(y_true, epsilon, 1)
    y_pred = np.clip(y_pred, epsilon, 1)
    loss = np.sum(y_true * np.log(y_true / y_pred), axis=-1)
    return np.mean(loss)

def kl_divergence_backward(y_true, y_pred, epsilon=1e-15):
    """
    Computes the gradient of KL Divergence Loss with respect to y_pred.
    """
    y_true = np.clip(y_true, epsilon, 1)
    y_pred = np.clip(y_pred, epsilon, 1)
    # The gradient of P log(P/Q) = P(log P - log Q) wrt Q is -P/Q
    # Since we mean over the batch, we divide by batch size
    batch_size = y_true.shape[0] if y_true.ndim > 1 else 1
    grad = - (y_true / y_pred) / batch_size
    return grad

def test_component():
    print("Testing KL Divergence Loss Component...")
    # True distributions (e.g., softmax outputs)
    y_true = np.array([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]])
    # Predicted distributions
    y_pred = np.array([[0.1, 0.6, 0.3], [0.2, 0.7, 0.1]])

    loss = kl_divergence_forward(y_true, y_pred)
    grad = kl_divergence_backward(y_true, y_pred)

    print(f"KL Divergence Loss: {loss:.4f}")
    assert loss >= 0, "Loss should be non-negative"
    assert grad.shape == y_pred.shape, "Grad shape mismatch"

    # Check if closer distributions have lower loss
    y_pred_close = np.array([[0.19, 0.51, 0.30], [0.11, 0.79, 0.10]])
    loss_close = kl_divergence_forward(y_true, y_pred_close)
    print(f"KL Divergence Loss (closer): {loss_close:.4f}")
    assert loss_close < loss, "Closer distributions should have lower KL divergence"

    print("Tests passed.")

if __name__ == "__main__":
    test_component()
