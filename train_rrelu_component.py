import numpy as np
import json

class RReLU:
    def __init__(self, lower=0.125, upper=0.333, training=True):
        self.lower = lower
        self.upper = upper
        self.training = training
        self.alpha = None
        self.x = None

    def forward(self, x):
        self.x = x
        if self.training:
            # Sample alpha for each element where x < 0
            self.alpha = np.random.uniform(self.lower, self.upper, size=x.shape)
            out = np.where(x >= 0, x, x * self.alpha)
        else:
            alpha = (self.lower + self.upper) / 2.0
            out = np.where(x >= 0, x, x * alpha)
        return out

    def backward(self, grad_output):
        if self.training:
            grad_x = grad_output * np.where(self.x >= 0, 1.0, self.alpha)
        else:
            alpha = (self.lower + self.upper) / 2.0
            grad_x = grad_output * np.where(self.x >= 0, 1.0, alpha)
        return grad_x

def test_component():
    np.random.seed(42)
    x = np.random.randn(10, 5)
    rrelu = RReLU(lower=0.1, upper=0.3, training=True)
    out = rrelu.forward(x)

    grad_output = np.ones_like(x)
    grad_x = rrelu.backward(grad_output)

    # Simple MSE to check learning is not really applicable as this is just an activation,
    # but we can do a dummy regression to show it trains

    # Let's train a simple linear model with RReLU on XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))

    rrelu_act = RReLU(lower=0.1, upper=0.3, training=True)

    learning_rate = 0.1
    epochs = 1000

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = rrelu_act.forward(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = z2

        # Loss
        loss = np.mean((y_pred - y) ** 2)

        # Backward pass
        grad_y_pred = 2 * (y_pred - y) / y.size
        grad_W2 = np.dot(a1.T, grad_y_pred)
        grad_b2 = np.sum(grad_y_pred, axis=0, keepdims=True)

        grad_a1 = np.dot(grad_y_pred, W2.T)
        grad_z1 = rrelu_act.backward(grad_a1)

        grad_W1 = np.dot(X.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        # Update weights
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2

    print(f"Final loss: {loss}")
    return loss < 0.1

if __name__ == '__main__':
    success = test_component()
    print("Success" if success else "Failed")
