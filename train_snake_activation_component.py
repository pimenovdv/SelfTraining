import numpy as np

def snake_activation(x, a=1.0):
    return x + (1.0 / a) * np.sin(a * x)**2

def snake_derivative(x, a=1.0):
    return 1.0 + np.sin(2 * a * x)

class SnakeLayer:
    def __init__(self, a=1.0):
        self.a = a
        self.input = None

    def forward(self, x):
        self.input = x
        return snake_activation(x, self.a)

    def backward(self, grad_output):
        return grad_output * snake_derivative(self.input, self.a)

def test_snake_activation():
    np.random.seed(42)
    # Simple test to make sure it runs and learns
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))

    activation = SnakeLayer(a=1.0)

    learning_rate = 0.1
    epochs = 1000

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = activation.forward(z1)
        z2 = np.dot(a1, W2) + b2

        # Mean Squared Error
        loss = np.mean((z2 - Y) ** 2)

        # Backward pass
        dz2 = 2 * (z2 - Y) / Y.size
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = activation.backward(da1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    print(f"Final Loss: {loss:.4f}")
    assert loss < 0.1, "Model did not converge"
    print("Snake Activation component test passed!")

if __name__ == "__main__":
    test_snake_activation()
