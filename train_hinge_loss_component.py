import numpy as np

def hinge_loss(y_true, y_pred):
    """
    Computes the hinge loss.
    y_true: expected to be 1 or -1.
    y_pred: raw scores (logits).
    """
    loss = np.maximum(0, 1 - y_true * y_pred)
    return np.mean(loss)

def hinge_loss_gradient(y_true, y_pred):
    """
    Computes gradient of hinge loss w.r.t y_pred.
    """
    grad = np.where(y_true * y_pred < 1, -y_true, 0.0)
    return grad / y_true.size

class LinearSVM:
    def __init__(self, input_dim):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def forward(self, X):
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y, epochs=1000, lr=0.01):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = hinge_loss(y, y_pred)
            losses.append(loss)

            grad_y_pred = hinge_loss_gradient(y, y_pred)

            # Gradients for weights and bias
            grad_w = np.dot(X.T, grad_y_pred)
            grad_b = np.sum(grad_y_pred)

            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return losses

def test_hinge_loss():
    print("Testing Hinge Loss Component...")

    np.random.seed(42)
    # Class 1
    X_pos = np.random.randn(50, 2) + 2.0
    y_pos = np.ones(50)

    # Class -1
    X_neg = np.random.randn(50, 2) - 2.0
    y_neg = -np.ones(50)

    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))

    model = LinearSVM(input_dim=2)

    print("Training Linear SVM with Hinge Loss:")
    model.train(X, y, epochs=500, lr=0.1)

    y_pred = model.forward(X)
    final_loss = hinge_loss(y, y_pred)
    print(f"Final Loss: {final_loss:.4f}")

    if final_loss < 0.1:
        print("Test passed successfully.")
    else:
        print("Test failed. Loss is too high.")

if __name__ == "__main__":
    test_hinge_loss()
