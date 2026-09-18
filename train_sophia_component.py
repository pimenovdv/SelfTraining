import numpy as np

def sophia_update(w, grad, m, h, lr, beta1, beta2, rho, weight_decay, t):
    if weight_decay > 0:
        grad = grad + weight_decay * w
    m = beta1 * m + (1 - beta1) * grad
    h = beta2 * h + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    h_hat = h / (1 - beta2 ** t)
    step = m_hat / (h_hat + 1e-8)
    step_clipped = np.clip(step, -rho, rho)
    w_new = w - lr * step_clipped
    return w_new, m, h

def test_sophia_component():
    print("Testing Sophia Optimizer Component...")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    np.random.seed(42)
    input_size = 2
    hidden_size = 8
    output_size = 1

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((1, output_size))

    lr = 0.5
    beta1 = 0.965
    beta2 = 0.99
    rho = 0.04
    weight_decay = 0.0

    m_W1 = np.zeros_like(W1)
    m_b1 = np.zeros_like(b1)
    m_W2 = np.zeros_like(W2)
    m_b2 = np.zeros_like(b2)

    h_W1 = np.zeros_like(W1)
    h_b1 = np.zeros_like(b1)
    h_W2 = np.zeros_like(W2)
    h_b2 = np.zeros_like(b2)

    epochs = 2000
    for epoch in range(epochs):
        z1 = np.dot(X, W1) + b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, W2) + b2
        a2 = 1 / (1 + np.exp(-z2))

        loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))

        dz2 = a2 - y
        dW2 = np.dot(a1.T, dz2) / X.shape[0]
        db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * (z1 > 0)
        dW1 = np.dot(X.T, dz1) / X.shape[0]
        db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]

        t = epoch + 1
        W1, m_W1, h_W1 = sophia_update(W1, dW1, m_W1, h_W1, lr, beta1, beta2, rho, weight_decay, t)
        b1, m_b1, h_b1 = sophia_update(b1, db1, m_b1, h_b1, lr, beta1, beta2, rho, weight_decay, t)
        W2, m_W2, h_W2 = sophia_update(W2, dW2, m_W2, h_W2, lr, beta1, beta2, rho, weight_decay, t)
        b2, m_b2, h_b2 = sophia_update(b2, db2, m_b2, h_b2, lr, beta1, beta2, rho, weight_decay, t)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    final_z1 = np.dot(X, W1) + b1
    final_a1 = np.maximum(0, final_z1)
    final_z2 = np.dot(final_a1, W2) + b2
    final_a2 = 1 / (1 + np.exp(-final_z2))

    print(f"Final Loss: {loss:.4f}")
    print(f"Predictions:\n{final_a2}")
    assert loss < 0.2, "Sophia Optimizer failed to converge on XOR problem"
    print("Sophia Optimizer Component test passed!")

if __name__ == "__main__":
    test_sophia_component()
