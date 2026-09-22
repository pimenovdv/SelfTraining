import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_asgd():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = 2
    hidden_size = 4
    output_size = 1

    epochs = 100000
    learning_rate = 5.0
    t0 = 100 # Averaging start time

    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 1.0
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 1.0
    b2 = np.zeros((1, output_size))

    avg_W1 = np.copy(W1)
    avg_b1 = np.copy(b1)
    avg_W2 = np.copy(W2)
    avg_b2 = np.copy(b2)

    for epoch in range(1, epochs + 1):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Backward pass
        dZ2 = (a2 - y) * sigmoid_derivative(z2)
        dW2 = np.dot(a1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

        dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(z1)
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

        # SGD Update with decreasing learning rate
        lr = learning_rate / (1.0 + 0.001 * epoch)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # ASGD Averaging
        if epoch > t0:
            avg_W1 = (avg_W1 * (epoch - t0 - 1) + W1) / (epoch - t0)
            avg_b1 = (avg_b1 * (epoch - t0 - 1) + b1) / (epoch - t0)
            avg_W2 = (avg_W2 * (epoch - t0 - 1) + W2) / (epoch - t0)
            avg_b2 = (avg_b2 * (epoch - t0 - 1) + b2) / (epoch - t0)
        else:
            avg_W1 = np.copy(W1)
            avg_b1 = np.copy(b1)
            avg_W2 = np.copy(W2)
            avg_b2 = np.copy(b2)

    # Final evaluation with averaged weights
    z1 = np.dot(X, avg_W1) + avg_b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, avg_W2) + avg_b2
    a2 = sigmoid(z2)

    print("Final predictions:")
    print(a2)
    assert np.all(np.abs(a2 - y) < 0.1), "Model failed to learn XOR."
    print("Success")

if __name__ == "__main__":
    train_asgd()
