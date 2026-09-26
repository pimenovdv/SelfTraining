import numpy as np

def log_sigmoid(x):
    """Log-Sigmoid Activation Component"""
    # For numerical stability:
    # log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
    # If x is large negative, log(1 + exp(-x)) will overflow.
    # log(1 + exp(-x)) = log(exp(-x) * (exp(x) + 1)) = -x + log(1 + exp(x))
    # So we use np.where
    return np.where(x > 0, -np.log(1 + np.exp(-x)), x - np.log(1 + np.exp(x)))

def log_sigmoid_derivative(x):
    """Derivative of Log-Sigmoid Activation Component"""
    # d/dx log(sigmoid(x)) = 1 - sigmoid(x)
    return 1 - (1 / (1 + np.exp(-x)))

def test_log_sigmoid():
    np.random.seed(42)
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_dim = 2
    hidden_dim = 8
    output_dim = 1

    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))

    learning_rate = 0.5
    epochs = 10000

    for epoch in range(epochs):
        z1 = np.dot(X, W1) + b1
        a1 = log_sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = 1 / (1 + np.exp(-z2))

        loss = np.mean((a2 - y)**2)

        da2 = 2 * (a2 - y) / y.size
        dz2 = da2 * a2 * (1 - a2)

        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * log_sigmoid_derivative(z1)

        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    print(f"Final loss: {loss:.4f}")
    predictions = (a2 > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    if accuracy == 1.0:
        print("Model successfully learned XOR with Log-Sigmoid.")
    else:
        print("Model failed to learn XOR.")

if __name__ == '__main__':
    test_log_sigmoid()
