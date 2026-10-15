import numpy as np

class PseudoHuberLoss:
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_pred, y_true):
        r = y_pred - y_true
        return np.mean(self.delta**2 * (np.sqrt(1 + (r / self.delta)**2) - 1))

    def backward(self, y_pred, y_true):
        r = y_pred - y_true
        return (r / np.sqrt(1 + (r / self.delta)**2)) / y_pred.size

if __name__ == "__main__":
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.5, 5.0, 2.0])

    loss_fn = PseudoHuberLoss(delta=1.0)
    loss = loss_fn.forward(y_pred, y_true)
    grad = loss_fn.backward(y_pred, y_true)

    print(f"Loss: {loss:.4f}")
    print(f"Gradient: {grad}")

    # Numerical gradient check
    epsilon = 1e-5
    for i in range(len(y_pred)):
        y_pred_plus = y_pred.copy()
        y_pred_plus[i] += epsilon
        loss_plus = loss_fn.forward(y_pred_plus, y_true)

        y_pred_minus = y_pred.copy()
        y_pred_minus[i] -= epsilon
        loss_minus = loss_fn.forward(y_pred_minus, y_true)

        num_grad = (loss_plus - loss_minus) / (2 * epsilon)
        print(f"Numerical: {num_grad:.6f}, Analytical: {grad[i]:.6f}")
        assert np.abs(num_grad - grad[i]) < 1e-4

    print("Pseudo-Huber Loss mathematical verification passed.")
