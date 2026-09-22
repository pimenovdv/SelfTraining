import numpy as np

class NLLLoss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        """
        y_pred: (batch_size, num_classes) - Log-probabilities of classes
        y_true: (batch_size,) - Integer class labels (0 to num_classes-1)
        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = y_pred.shape[0]

        # Gather the log probabilities of the true classes
        loss = -np.sum(y_pred[np.arange(self.batch_size), y_true]) / self.batch_size
        return loss

    def backward(self):
        """
        Computes the gradient of NLL Loss w.r.t input.
        """
        grad = np.zeros_like(self.y_pred)
        # Gradient is -1/N for the true class, 0 otherwise
        grad[np.arange(self.batch_size), self.y_true] = -1.0 / self.batch_size
        return grad

def test_nll_loss():
    print("Testing Negative Log Likelihood (NLL) Loss...")

    # Simulate a network output (log probabilities, e.g. output of LogSoftmax)
    # Batch size of 3, 4 classes
    y_pred = np.array([
        [-0.1, -2.0, -3.0, -4.0],
        [-1.0, -0.5, -2.0, -1.5],
        [-3.0, -2.0, -0.2, -1.0]
    ])

    # Ground truth class indices
    y_true = np.array([0, 1, 2])

    nll = NLLLoss()

    # Test forward pass
    loss = nll.forward(y_pred, y_true)
    expected_loss = -(-0.1 - 0.5 - 0.2) / 3
    print(f"Forward Loss: {loss:.4f} (Expected: {expected_loss:.4f})")
    assert np.isclose(loss, expected_loss), "Forward pass failed"

    # Test backward pass
    grad = nll.backward()
    expected_grad = np.array([
        [-1/3, 0, 0, 0],
        [0, -1/3, 0, 0],
        [0, 0, -1/3, 0]
    ])
    print("Backward Gradient:\n", grad)
    assert np.allclose(grad, expected_grad), "Backward pass failed"

    print("NLL Loss component successfully implemented and tested!")

if __name__ == "__main__":
    test_nll_loss()
