import numpy as np

def xavier_initialize(fan_in, fan_out):
    """
    Xavier (Glorot) initialization for weights.
    Draws weights from a normal distribution with mean 0 and variance 2.0 / (fan_in + fan_out).
    """
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def test_xavier_initialization():
    np.random.seed(42)

    # Simple dataset: XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = 2
    hidden_size = 4
    output_size = 1

    # Initialize with Xavier
    W1 = xavier_initialize(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = xavier_initialize(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    learning_rate = 0.5
    epochs = 4000

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = tanh(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Loss
        loss = mse_loss(y, a2)

        # Backward pass
        da2 = mse_loss_derivative(y, a2)
        dz2 = da2 * sigmoid_derivative(z2)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * tanh_derivative(z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    print(f"Final loss with Xavier Initialization: {loss:.6f}")
    predictions = (a2 > 0.5).astype(int)
    print(f"Predictions:\n{predictions}")

    assert loss < 0.1, "Model failed to converge with Xavier initialization."
    assert np.array_equal(predictions, y), "Model failed to solve XOR."
    print("Xavier Initialization component test passed successfully.")

if __name__ == "__main__":
    test_xavier_initialization()
