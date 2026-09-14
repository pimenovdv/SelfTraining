import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_rprop():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    input_size = 2
    hidden_size = 4
    output_size = 1

    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    # Rprop specific variables
    delta_W1 = np.full_like(W1, 0.1)
    delta_b1 = np.full_like(b1, 0.1)
    delta_W2 = np.full_like(W2, 0.1)
    delta_b2 = np.full_like(b2, 0.1)

    prev_grad_W1 = np.zeros_like(W1)
    prev_grad_b1 = np.zeros_like(b1)
    prev_grad_W2 = np.zeros_like(W2)
    prev_grad_b2 = np.zeros_like(b2)

    eta_plus = 1.2
    eta_minus = 0.5
    delta_max = 50.0
    delta_min = 1e-6

    epochs = 1000
    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Loss (MSE)
        loss = np.mean((a2 - Y) ** 2)

        # Backward pass
        error = a2 - Y
        d_a2 = error * sigmoid_derivative(z2)
        grad_W2 = np.dot(a1.T, d_a2) / Y.size
        grad_b2 = np.sum(d_a2, axis=0, keepdims=True) / Y.size

        d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(z1)
        grad_W1 = np.dot(X.T, d_a1) / Y.size
        grad_b1 = np.sum(d_a1, axis=0, keepdims=True) / Y.size

        # Rprop Update
        for param, grad, prev_grad, delta in [
            (W1, grad_W1, prev_grad_W1, delta_W1),
            (b1, grad_b1, prev_grad_b1, delta_b1),
            (W2, grad_W2, prev_grad_W2, delta_W2),
            (b2, grad_b2, prev_grad_b2, delta_b2)
        ]:
            change = grad * prev_grad

            # Where gradient sign hasn't changed
            pos_mask = change > 0
            delta[pos_mask] = np.minimum(delta[pos_mask] * eta_plus, delta_max)
            param[pos_mask] -= np.sign(grad[pos_mask]) * delta[pos_mask]
            prev_grad[pos_mask] = grad[pos_mask]

            # Where gradient sign changed
            neg_mask = change < 0
            delta[neg_mask] = np.maximum(delta[neg_mask] * eta_minus, delta_min)
            prev_grad[neg_mask] = 0.0 # prevent double penalty

            # Where gradient is exactly 0
            zero_mask = change == 0
            param[zero_mask] -= np.sign(grad[zero_mask]) * delta[zero_mask]
            prev_grad[zero_mask] = grad[zero_mask]

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Final Predictions:")
    print(a2)
    assert loss < 0.1, "Rprop optimizer failed to converge"
    print("Rprop Optimizer component test passed!")

if __name__ == "__main__":
    train_rprop()
