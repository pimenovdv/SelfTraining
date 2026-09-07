import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_adadelta():
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

    gamma = 0.95
    epsilon = 1e-6

    # Exponentially decaying average of squared gradients
    E_g2_W1 = np.zeros_like(W1)
    E_g2_b1 = np.zeros_like(b1)
    E_g2_W2 = np.zeros_like(W2)
    E_g2_b2 = np.zeros_like(b2)

    # Exponentially decaying average of squared updates
    E_dx2_W1 = np.zeros_like(W1)
    E_dx2_b1 = np.zeros_like(b1)
    E_dx2_W2 = np.zeros_like(W2)
    E_dx2_b2 = np.zeros_like(b2)

    epochs = 15000
    for epoch in range(epochs):
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

        # Adadelta Updates
        E_g2_W2 = gamma * E_g2_W2 + (1 - gamma) * (d_W2 ** 2)
        E_g2_b2 = gamma * E_g2_b2 + (1 - gamma) * (d_b2 ** 2)
        E_g2_W1 = gamma * E_g2_W1 + (1 - gamma) * (d_W1 ** 2)
        E_g2_b1 = gamma * E_g2_b1 + (1 - gamma) * (d_b1 ** 2)

        rms_dx_W2 = np.sqrt(E_dx2_W2 + epsilon)
        rms_dx_b2 = np.sqrt(E_dx2_b2 + epsilon)
        rms_dx_W1 = np.sqrt(E_dx2_W1 + epsilon)
        rms_dx_b1 = np.sqrt(E_dx2_b1 + epsilon)

        rms_g_W2 = np.sqrt(E_g2_W2 + epsilon)
        rms_g_b2 = np.sqrt(E_g2_b2 + epsilon)
        rms_g_W1 = np.sqrt(E_g2_W1 + epsilon)
        rms_g_b1 = np.sqrt(E_g2_b1 + epsilon)

        dx_W2 = - (rms_dx_W2 / rms_g_W2) * d_W2
        dx_b2 = - (rms_dx_b2 / rms_g_b2) * d_b2
        dx_W1 = - (rms_dx_W1 / rms_g_W1) * d_W1
        dx_b1 = - (rms_dx_b1 / rms_g_b1) * d_b1

        E_dx2_W2 = gamma * E_dx2_W2 + (1 - gamma) * (dx_W2 ** 2)
        E_dx2_b2 = gamma * E_dx2_b2 + (1 - gamma) * (dx_b2 ** 2)
        E_dx2_W1 = gamma * E_dx2_W1 + (1 - gamma) * (dx_W1 ** 2)
        E_dx2_b1 = gamma * E_dx2_b1 + (1 - gamma) * (dx_b1 ** 2)

        W2 += dx_W2
        b2 += dx_b2
        W1 += dx_W1
        b1 += dx_b1

    # Evaluate
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    predictions = (a2 > 0.5).astype(int)
    accuracy = np.mean(predictions == Y)

    print(f"Final Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    if accuracy == 1.0:
        print("Success: Adadelta successfully learned the XOR mapping.")
    else:
        print("Failure: Adadelta failed to learn the XOR mapping.")
        exit(1)

if __name__ == "__main__":
    train_adadelta()
