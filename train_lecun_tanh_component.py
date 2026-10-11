import numpy as np

class LeCunTanh:
    def __init__(self):
        pass

    def forward(self, x):
        return 1.7159 * np.tanh(2/3 * x)

    def backward(self, x, grad_output):
        # d/dx (1.7159 * tanh(2/3 * x)) = 1.7159 * (2/3) * (1 - tanh^2(2/3 * x))
        # = 1.143933 * (1 - tanh^2(2/3 * x))
        tanh_val = np.tanh(2/3 * x)
        grad = 1.7159 * (2/3) * (1 - tanh_val**2)
        return grad_output * grad

def test_lecun_tanh():
    activation = LeCunTanh()
    x = np.array([-1.0, 0.0, 1.0])

    # Forward pass
    y = activation.forward(x)
    expected_y = 1.7159 * np.tanh(2/3 * x)
    assert np.allclose(y, expected_y), f"Forward failed: {y} != {expected_y}"

    # Backward pass
    grad_output = np.ones_like(x)
    grad = activation.backward(x, grad_output)
    expected_grad = 1.7159 * (2/3) * (1 - np.tanh(2/3 * x)**2)
    assert np.allclose(grad, expected_grad), f"Backward failed: {grad} != {expected_grad}"

    print("LeCun Tanh mathematical validation passed!")

def train_model():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    Y = np.sum(X, axis=1, keepdims=True) > 0 # Binary classification
    Y = Y.astype(float)

    # Simple 1-layer network
    W1 = np.random.randn(10, 1) * 0.1
    b1 = np.zeros((1, 1))

    activation = LeCunTanh()
    learning_rate = 0.01

    print("Starting training...")
    for epoch in range(100):
        # Forward
        z1 = np.dot(X, W1) + b1
        a1 = activation.forward(z1)

        # Loss (MSE)
        loss = np.mean((a1 - Y)**2)

        # Backward
        grad_a1 = 2 * (a1 - Y) / len(Y)
        grad_z1 = activation.backward(z1, grad_a1)

        grad_W1 = np.dot(X.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        # Update
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    test_lecun_tanh()
    train_model()
