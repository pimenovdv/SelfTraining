import numpy as np

def amsgrad_step(params, grads, m, v, v_hat, t, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    t += 1
    new_params = []
    new_m = []
    new_v = []
    new_v_hat = []

    for p, g, mi, vi, vhi in zip(params, grads, m, v, v_hat):
        m_t = beta1 * mi + (1 - beta1) * g
        v_t = beta2 * vi + (1 - beta2) * (g ** 2)
        v_hat_t = np.maximum(vhi, v_t)

        m_hat = m_t / (1 - beta1 ** t)
        v_hat_corr = v_hat_t / (1 - beta2 ** t)

        p_t = p - lr * m_hat / (np.sqrt(v_hat_corr) + epsilon)

        new_params.append(p_t)
        new_m.append(m_t)
        new_v.append(v_t)
        new_v_hat.append(v_hat_t)

    return new_params, new_m, new_v, new_v_hat, t

def test_amsgrad():
    np.random.seed(42)
    # Simple linear regression: y = 3x + 2
    X = np.random.randn(100, 1)
    y = 3 * X + 2 + np.random.randn(100, 1) * 0.1

    W = np.random.randn(1, 1)
    b = np.random.randn(1)

    params = [W, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    v_hat = [np.zeros_like(p) for p in params]
    t = 0

    lr = 0.1
    epochs = 100

    for epoch in range(epochs):
        # Forward
        predictions = X @ params[0] + params[1]
        loss = np.mean((predictions - y) ** 2)

        # Gradients
        dW = (2 / len(X)) * X.T @ (predictions - y)
        db = (2 / len(X)) * np.sum(predictions - y)
        grads = [dW, db]

        # Update
        params, m, v, v_hat, t = amsgrad_step(params, grads, m, v, v_hat, t, lr=lr)

    print(f"Final W: {params[0][0, 0]:.4f}, b: {params[1][0]:.4f}, loss: {loss:.4f}")
    assert loss < 0.1, "Loss should be low for linear regression."
    print("AMSGrad test passed!")

if __name__ == "__main__":
    test_amsgrad()
