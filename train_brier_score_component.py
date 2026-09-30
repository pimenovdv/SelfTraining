import numpy as np

def brier_score_loss(y_true, y_pred):
    """
    Computes the Brier Score for predicted probabilities.
    Brier Score = 1/N * sum (y_pred - y_true)^2
    """
    return np.mean((y_pred - y_true)**2)

def brier_score_loss_derivative(y_true, y_pred):
    """
    Computes the derivative of the Brier Score with respect to y_pred.
    derivative = 2/N * (y_pred - y_true)
    """
    return 2.0 / y_true.size * (y_pred - y_true)

def test_brier_score_loss():
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.3])

    loss = brier_score_loss(y_true, y_pred)
    derivative = brier_score_loss_derivative(y_true, y_pred)

    print("Brier Score Loss Component")
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"Brier Score Loss: {loss:.4f}")
    print(f"Derivative w.r.t y_pred:\n{derivative}")

if __name__ == "__main__":
    test_brier_score_loss()
