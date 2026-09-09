import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_adamax():
    np.random.seed(42)
    # XOR dataset
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([[0], [1], [1], [0]])

    input_size = 2
    hidden_size = 4
    output_size = 1

    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    learning_rate = 0.05
    beta1 = 0.9
    beta2 = 0.999

    m_W1 = np.zeros_like(W1)
    m_b1 = np.zeros_like(b1)
    m_W2 = np.zeros_like(W2)
    m_b2 = np.zeros_like(b2)

    u_W1 = np.zeros_like(W1)
    u_b1 = np.zeros_like(b1)
    u_W2 = np.zeros_like(W2)
    u_b2 = np.zeros_like(b2)

    epochs = 15000
    for epoch in range(1, epochs + 1):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Loss (MSE)
        loss = np.mean(0.5 * (a2 - Y) ** 2)

        # Backward pass
        d_loss = a2 - Y
        d_z2 = d_loss * sigmoid_derivative(z2)
        d_W2 = np.dot(a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, W2.T)
        d_z1 = d_a1 * sigmoid_derivative(z1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        # Adamax Updates
        m_W2 = beta1 * m_W2 + (1 - beta1) * d_W2
        m_b2 = beta1 * m_b2 + (1 - beta1) * d_b2
        m_W1 = beta1 * m_W1 + (1 - beta1) * d_W1
        m_b1 = beta1 * m_b1 + (1 - beta1) * d_b1

        u_W2 = np.maximum(beta2 * u_W2, np.abs(d_W2))
        u_b2 = np.maximum(beta2 * u_b2, np.abs(d_b2))
        u_W1 = np.maximum(beta2 * u_W1, np.abs(d_W1))
        u_b1 = np.maximum(beta2 * u_b1, np.abs(d_b1))

        m_hat_W2 = m_W2 / (1 - beta1 ** epoch)
        m_hat_b2 = m_b2 / (1 - beta1 ** epoch)
        m_hat_W1 = m_W1 / (1 - beta1 ** epoch)
        m_hat_b1 = m_b1 / (1 - beta1 ** epoch)

        W2 -= (learning_rate / (u_W2 + 1e-8)) * m_hat_W2
        b2 -= (learning_rate / (u_b2 + 1e-8)) * m_hat_b2
        W1 -= (learning_rate / (u_W1 + 1e-8)) * m_hat_W1
        b1 -= (learning_rate / (u_b1 + 1e-8)) * m_hat_b1

    # Evaluate
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    predictions = (a2 > 0.5).astype(int)
    accuracy = np.mean(predictions == Y)

    print(f"Final Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}")

    if accuracy == 1.0:
        print("Success: Adamax successfully learned the XOR mapping.")
    else:
        print("Failure: Adamax failed to learn the XOR mapping.")
        exit(1)

if __name__ == "__main__":
    train_adamax()
