import numpy as np
import json
import os

def welsch_loss(y_true, y_pred, c=1.0):
    r"""
    Computes the Welsch loss.
    L(y, \hat{y}) = 1 - exp(- (y - \hat{y})^2 / (2 * c^2))
    """
    diff = y_true - y_pred
    loss = 1 - np.exp(- (diff ** 2) / (2 * c ** 2))
    return np.mean(loss)

def welsch_loss_derivative(y_true, y_pred, c=1.0):
    r"""
    Computes the derivative of the Welsch loss with respect to y_pred.
    d L / d \hat{y} = - (y - \hat{y}) / c^2 * exp(- (y - \hat{y})^2 / (2 * c^2))
    """
    diff = y_true - y_pred
    grad = - (diff / (c ** 2)) * np.exp(- (diff ** 2) / (2 * c ** 2))
    return grad / y_true.size

def test_welsch_loss():
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = np.random.randn(100)

    loss = welsch_loss(y_true, y_pred, c=1.5)
    grad = welsch_loss_derivative(y_true, y_pred, c=1.5)

    print(f"Welsch Loss: {loss:.4f}")
    print(f"Gradient sum: {np.sum(grad):.4f}")
    print(f"Gradient var: {np.var(grad):.4f}")

    os.makedirs('results', exist_ok=True)
    with open('results/welsch_loss_results.json', 'w') as f:
        json.dump({
            "welsch_loss": float(loss),
            "gradient_sum": float(np.sum(grad)),
            "gradient_var": float(np.var(grad))
        }, f, indent=4)

if __name__ == "__main__":
    test_welsch_loss()
