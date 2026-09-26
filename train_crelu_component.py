import numpy as np

def crelu(x):
    """Concatenated ReLU"""
    return np.concatenate((np.maximum(0, x), np.maximum(0, -x)), axis=-1)

def crelu_derivative(x):
    """Derivative of CReLU"""
    grad_pos = (x > 0).astype(float)
    grad_neg = (-x > 0).astype(float) * -1.0 # The output was max(0, -x). The derivative w.r.t. x is -1 when -x > 0
    return np.concatenate((grad_pos, grad_neg), axis=-1)

def test_crelu():
    np.random.seed(42)
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Simple 2-layer neural network with CReLU
    input_dim = 2
    hidden_dim = 4
    output_dim = 1

    # In CReLU, the hidden dimension output will be hidden_dim * 2
    crelu_dim = hidden_dim * 2

    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(crelu_dim, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))

    learning_rate = 0.1
    epochs = 10000

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = crelu(z1)
        z2 = np.dot(a1, W2) + b2
        # Use sigmoid for final layer
        a2 = 1 / (1 + np.exp(-z2))

        # Loss (MSE)
        loss = np.mean((a2 - y)**2)

        # Backward pass
        # MSE derivative w.r.t a2: 2 * (a2 - y) / N
        da2 = 2 * (a2 - y) / y.size
        # Sigmoid derivative: a2 * (1 - a2)
        dz2 = da2 * a2 * (1 - a2)

        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * crelu_derivative(z1)
        # Note: dz1 has shape (N, crelu_dim). We need to sum gradients for the original hidden_dim.
        # Since a1 = [relu(z1), relu(-z1)], da1 has shape (N, hidden_dim * 2).
        # dz1 (pos part) = da1[:, :hidden_dim] * (z1 > 0)
        # dz1 (neg part) = da1[:, hidden_dim:] * (-1) * (-z1 > 0)
        # The total gradient w.r.t z1 is the sum of these two parts
        dz1_total = dz1[:, :hidden_dim] + dz1[:, hidden_dim:]

        dW1 = np.dot(X.T, dz1_total)
        db1 = np.sum(dz1_total, axis=0, keepdims=True)

        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    print(f"Final loss: {loss:.4f}")
    predictions = (a2 > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    if accuracy == 1.0:
        print("Model successfully learned XOR with CReLU.")
    else:
        print("Model failed to learn XOR.")

if __name__ == '__main__':
    test_crelu()
