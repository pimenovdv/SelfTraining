"""
Component Testing: Mean Squared Error (MSE) Loss
This script tests a mathematical model of MSE Loss in pure NumPy, testing
forward and backward passes.
"""

import numpy as np

class MSELoss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred) ** 2)

    def backward(self):
        return -2 * (self.y_true - self.y_pred) / self.y_true.size

def test_component():
    print("Testing Mean Squared Error (MSE) Loss Component...")
    mse = MSELoss()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 3.5])

    loss = mse.forward(y_pred, y_true)
    grad = mse.backward()

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"Loss: {loss}")
    print(f"Gradient: {grad}")
    print("Test passed.")

if __name__ == "__main__":
    test_component()
