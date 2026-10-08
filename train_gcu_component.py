import numpy as np

def gcu(x):
    """
    Growing Cosine Unit (GCU) activation function.
    GCU(x) = x * np.cos(x)
    """
    return x * np.cos(x)

def gcu_derivative(x):
    """
    Derivative of GCU.
    d/dx (x * cos(x)) = cos(x) - x * sin(x)
    """
    return np.cos(x) - x * np.sin(x)

def test_gcu():
    x = np.array([-np.pi, 0.0, np.pi])
    expected = x * np.cos(x)
    y = gcu(x)
    assert np.allclose(y, expected), "GCU forward failed"

    grad = gcu_derivative(x)
    expected_grad = np.cos(x) - x * np.sin(x)
    assert np.allclose(grad, expected_grad), "GCU backward failed"
    print("GCU mathematical validation passed!")

def train_model():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    Y = np.sum(X, axis=1, keepdims=True) + np.random.randn(100, 1) * 0.1

    W1 = np.random.randn(10, 20) * 0.1
    b1 = np.zeros((1, 20))
    W2 = np.random.randn(20, 1) * 0.1
    b2 = np.zeros((1, 1))

    lr = 0.01
    print("Starting training...")
    for epoch in range(100):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = gcu(z1)
        z2 = np.dot(a1, W2) + b2
        output = z2

        loss = np.mean((output - Y) ** 2)

        # Backward pass
        d_output = 2 * (output - Y) / X.shape[0]

        dW2 = np.dot(a1.T, d_output)
        db2 = np.sum(d_output, axis=0, keepdims=True)

        d_a1 = np.dot(d_output, W2.T)
        d_z1 = d_a1 * gcu_derivative(z1)

        dW1 = np.dot(X.T, d_z1)
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    test_gcu()
    train_model()
