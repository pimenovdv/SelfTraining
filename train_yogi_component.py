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

def test_yogi_component():
    np.random.seed(42)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Network architecture
    input_size = 2
    hidden_size = 4
    output_size = 1

    # Initialize weights
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))

    # Yogi hyperparameters
    learning_rate = 0.05
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Initialize Yogi moments
    m_W1 = np.zeros_like(W1)
    v_W1 = np.zeros_like(W1)
    m_b1 = np.zeros_like(b1)
    v_b1 = np.zeros_like(b1)

    m_W2 = np.zeros_like(W2)
    v_W2 = np.zeros_like(W2)
    m_b2 = np.zeros_like(b2)
    v_b2 = np.zeros_like(b2)

    epochs = 4000
    for t in range(1, epochs + 1):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Loss
        loss = bce_loss(y, a2)

        # Backward pass
        dz2 = a2 - y
        dW2 = np.dot(a1.T, dz2) / X.shape[0]
        db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * sigmoid_derivative(z1)
        dW1 = np.dot(X.T, dz1) / X.shape[0]
        db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]

        # Yogi update for each parameter
        params = [(W1, dW1, m_W1, v_W1), (b1, db1, m_b1, v_b1),
                  (W2, dW2, m_W2, v_W2), (b2, db2, m_b2, v_b2)]

        for param, grad, m, v in params:
            m[:] = beta1 * m + (1 - beta1) * grad

            grad_sq = grad ** 2
            v[:] = v - (1 - beta2) * np.sign(v - grad_sq) * grad_sq

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            param[:] = param - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        if t % 1000 == 0:
            print(f"Epoch {t}, Loss: {loss:.4f}")

    print("Final Predictions:")
    print(np.round(a2, 3))

    assert loss < 0.1, "Loss did not converge!"
    print("Success: Yogi optimizer successfully trained the network on XOR.")

if __name__ == "__main__":
    test_yogi_component()
