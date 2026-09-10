import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_momentum():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    input_size = 2
    hidden_size = 4
    output_size = 1

    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    learning_rate = 0.5
    momentum = 0.9

    v_W1 = np.zeros_like(W1)
    v_b1 = np.zeros_like(b1)
    v_W2 = np.zeros_like(W2)
    v_b2 = np.zeros_like(b2)

    epochs = 10000
    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Loss (MSE)
        loss = np.mean((a2 - y) ** 2)

        # Backward pass
        d_a2 = 2 * (a2 - y) / y.size
        d_z2 = d_a2 * sigmoid_derivative(z2)
        d_W2 = np.dot(a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, W2.T)
        d_z1 = d_a1 * sigmoid_derivative(z1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        # Momentum updates
        v_W1 = momentum * v_W1 + learning_rate * d_W1
        v_b1 = momentum * v_b1 + learning_rate * d_b1
        v_W2 = momentum * v_W2 + learning_rate * d_W2
        v_b2 = momentum * v_b2 + learning_rate * d_b2

        W1 -= v_W1
        b1 -= v_b1
        W2 -= v_W2
        b2 -= v_b2

    print(f"Final Loss: {loss}")
    predictions = (a2 > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100}%")

    if accuracy == 1.0 and loss < 0.1:
        print("Success")
    else:
        print("Failure")
        exit(1)

if __name__ == "__main__":
    train_momentum()
