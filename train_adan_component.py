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

def train_adan_optimizer():
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

    lr = 0.05
    beta1 = 0.98
    beta2 = 0.92
    beta3 = 0.99
    eps = 1e-8
    weight_decay = 0.02

    m_W1 = np.zeros_like(W1)
    v_W1 = np.zeros_like(W1)
    n_W1 = np.zeros_like(W1)
    prev_g_W1 = np.zeros_like(W1)

    m_b1 = np.zeros_like(b1)
    v_b1 = np.zeros_like(b1)
    n_b1 = np.zeros_like(b1)
    prev_g_b1 = np.zeros_like(b1)

    m_W2 = np.zeros_like(W2)
    v_W2 = np.zeros_like(W2)
    n_W2 = np.zeros_like(W2)
    prev_g_W2 = np.zeros_like(W2)

    m_b2 = np.zeros_like(b2)
    v_b2 = np.zeros_like(b2)
    n_b2 = np.zeros_like(b2)
    prev_g_b2 = np.zeros_like(b2)

    epochs = 5000
    for epoch in range(1, epochs + 1):
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

        # Adan updates
        def adan_update(param, grad, prev_grad, m, v, n):
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad - prev_grad)
            n = beta3 * n + (1 - beta3) * ((grad + (1 - beta2) * (grad - prev_grad)) ** 2)

            m_hat = m / (1 - beta1 ** epoch)
            v_hat = v / (1 - beta2 ** epoch)
            n_hat = n / (1 - beta3 ** epoch)

            update = m_hat + (1 - beta2) * v_hat
            param = param * (1 - lr * weight_decay) - lr * update / (np.sqrt(n_hat) + eps)
            return param, grad, m, v, n

        W1, prev_g_W1, m_W1, v_W1, n_W1 = adan_update(W1, dW1, prev_g_W1, m_W1, v_W1, n_W1)
        b1, prev_g_b1, m_b1, v_b1, n_b1 = adan_update(b1, db1, prev_g_b1, m_b1, v_b1, n_b1)
        W2, prev_g_W2, m_W2, v_W2, n_W2 = adan_update(W2, dW2, prev_g_W2, m_W2, v_W2, n_W2)
        b2, prev_g_b2, m_b2, v_b2, n_b2 = adan_update(b2, db2, prev_g_b2, m_b2, v_b2, n_b2)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Final prediction
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    predictions = sigmoid(z2)
    print("Final Predictions:")
    print(predictions)

if __name__ == "__main__":
    train_adan_optimizer()
