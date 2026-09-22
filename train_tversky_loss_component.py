import numpy as np

def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-6):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    true_pos = np.sum(y_true * y_pred)
    false_neg = np.sum(y_true * (1 - y_pred))
    false_pos = np.sum((1 - y_true) * y_pred)

    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    return 1 - tversky_index

def tversky_loss_grad(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-6):
    # Gradient of Tversky loss wrt y_pred
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    true_pos = np.sum(y_true * y_pred)
    false_neg = np.sum(y_true * (1 - y_pred))
    false_pos = np.sum((1 - y_true) * y_pred)

    num = true_pos + smooth
    den = true_pos + alpha * false_pos + beta * false_neg + smooth

    # dT/d(y_pred_i) = [ d(num)/dy_pred_i * den - num * d(den)/dy_pred_i ] / den^2
    # d(num)/dy_pred_i = y_true_i
    # d(den)/dy_pred_i = y_true_i + alpha * (1 - y_true_i) - beta * y_true_i
    #                  = y_true_i(1 - beta) + alpha(1 - y_true_i)

    d_num = y_true
    d_den = y_true * (1 - beta) + alpha * (1 - y_true)

    grad = (d_num * den - num * d_den) / (den ** 2)
    return -grad.reshape(y_true.shape)

def test_tversky_loss():
    np.random.seed(42)
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.9, 0.3, 0.1])

    # Forward
    loss = tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7)
    print(f"Tversky Loss: {loss:.4f}")

    # Backward
    grad = tversky_loss_grad(y_true, y_pred, alpha=0.3, beta=0.7)
    print(f"Gradient: {grad}")

    # Numerical gradient check
    epsilon = 1e-5
    numerical_grad = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        y_pred_plus = y_pred.copy()
        y_pred_plus[i] += epsilon
        loss_plus = tversky_loss(y_true, y_pred_plus, alpha=0.3, beta=0.7)

        y_pred_minus = y_pred.copy()
        y_pred_minus[i] -= epsilon
        loss_minus = tversky_loss(y_true, y_pred_minus, alpha=0.3, beta=0.7)

        numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

    print(f"Numerical Gradient: {numerical_grad}")

    np.testing.assert_allclose(grad, numerical_grad, rtol=1e-4, atol=1e-4)
    print("Gradient check passed!")

if __name__ == '__main__':
    test_tversky_loss()
