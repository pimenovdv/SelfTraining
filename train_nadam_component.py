import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def train():
    np.random.seed(42)
    # XOR dataset
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = 2
    hidden_size = 4
    output_size = 1

    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    # Nadam parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    lr = 0.1
    epochs = 10000

    m_W1 = np.zeros_like(W1)
    v_W1 = np.zeros_like(W1)
    m_b1 = np.zeros_like(b1)
    v_b1 = np.zeros_like(b1)
    m_W2 = np.zeros_like(W2)
    v_W2 = np.zeros_like(W2)
    m_b2 = np.zeros_like(b2)
    v_b2 = np.zeros_like(b2)

    for epoch in range(1, epochs + 1):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Backward pass
        dz2 = (a2 - y) * sigmoid_derivative(z2)
        dW2 = np.dot(a1.T, dz2) / X.shape[0]
        db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]

        dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(z1)
        dW1 = np.dot(X.T, dz1) / X.shape[0]
        db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]

        # Nadam updates
        # W1
        m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
        v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1 ** 2)
        m_W1_hat = m_W1 / (1 - beta1 ** epoch)
        v_W1_hat = v_W1 / (1 - beta2 ** epoch)
        dW1_hat = dW1 / (1 - beta1 ** epoch)
        m_W1_nesterov = beta1 * m_W1_hat + (1 - beta1) * dW1_hat
        W1 -= lr * m_W1_nesterov / (np.sqrt(v_W1_hat) + epsilon)

        # b1
        m_b1 = beta1 * m_b1 + (1 - beta1) * db1
        v_b1 = beta2 * v_b1 + (1 - beta2) * (db1 ** 2)
        m_b1_hat = m_b1 / (1 - beta1 ** epoch)
        v_b1_hat = v_b1 / (1 - beta2 ** epoch)
        db1_hat = db1 / (1 - beta1 ** epoch)
        m_b1_nesterov = beta1 * m_b1_hat + (1 - beta1) * db1_hat
        b1 -= lr * m_b1_nesterov / (np.sqrt(v_b1_hat) + epsilon)

        # W2
        m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
        v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2 ** 2)
        m_W2_hat = m_W2 / (1 - beta1 ** epoch)
        v_W2_hat = v_W2 / (1 - beta2 ** epoch)
        dW2_hat = dW2 / (1 - beta1 ** epoch)
        m_W2_nesterov = beta1 * m_W2_hat + (1 - beta1) * dW2_hat
        W2 -= lr * m_W2_nesterov / (np.sqrt(v_W2_hat) + epsilon)

        # b2
        m_b2 = beta1 * m_b2 + (1 - beta1) * db2
        v_b2 = beta2 * v_b2 + (1 - beta2) * (db2 ** 2)
        m_b2_hat = m_b2 / (1 - beta1 ** epoch)
        v_b2_hat = v_b2 / (1 - beta2 ** epoch)
        db2_hat = db2 / (1 - beta1 ** epoch)
        m_b2_nesterov = beta1 * m_b2_hat + (1 - beta1) * db2_hat
        b2 -= lr * m_b2_nesterov / (np.sqrt(v_b2_hat) + epsilon)

    print("Final predictions:")
    print(a2)
    assert np.all(np.abs(a2 - y) < 0.1), "Model failed to learn XOR."
    print("Success")

if __name__ == "__main__":
    train()
