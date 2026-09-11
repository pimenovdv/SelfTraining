import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def bce_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_lion_optimizer():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    input_size = 2
    hidden_size = 4
    output_size = 1

    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))

    lr = 0.01
    beta1 = 0.9
    beta2 = 0.99
    weight_decay = 0.01

    m_W1 = np.zeros_like(W1)
    m_b1 = np.zeros_like(b1)
    m_W2 = np.zeros_like(W2)
    m_b2 = np.zeros_like(b2)

    epochs = 10000
    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Backward pass
        loss = bce_loss(y, a2)
        dz2 = (a2 - y) / y.size
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * sigmoid_derivative(z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Lion update for W1
        c_W1 = beta1 * m_W1 + (1 - beta1) * dW1
        W1 -= lr * (np.sign(c_W1) + weight_decay * W1)
        m_W1 = beta2 * m_W1 + (1 - beta2) * dW1

        # Lion update for b1
        c_b1 = beta1 * m_b1 + (1 - beta1) * db1
        b1 -= lr * (np.sign(c_b1) + weight_decay * b1)
        m_b1 = beta2 * m_b1 + (1 - beta2) * db1

        # Lion update for W2
        c_W2 = beta1 * m_W2 + (1 - beta1) * dW2
        W2 -= lr * (np.sign(c_W2) + weight_decay * W2)
        m_W2 = beta2 * m_W2 + (1 - beta2) * dW2

        # Lion update for b2
        c_b2 = beta1 * m_b2 + (1 - beta1) * db2
        b2 -= lr * (np.sign(c_b2) + weight_decay * b2)
        m_b2 = beta2 * m_b2 + (1 - beta2) * db2

    print(f"Final Loss: {loss:.4f}")
    predictions = (a2 > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    if accuracy == 1.0:
        print("Success: Lion Optimizer converged and solved XOR.")
    else:
        print("Failure: Lion Optimizer did not solve XOR.")

if __name__ == "__main__":
    train_lion_optimizer()
