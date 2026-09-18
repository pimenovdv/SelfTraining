import numpy as np

def diffgrad_optimizer(params, grads, m, v, prev_g, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    diffGrad optimizer step.
    DiffGrad introduces a friction mechanism to Adam that slows down updates when the gradient changes rapidly.
    """
    m_new = beta1 * m + (1 - beta1) * grads
    v_new = beta2 * v + (1 - beta2) * (grads ** 2)

    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # diffGrad Friction Coefficient (DFC)
    diff = np.abs(prev_g - grads)
    dfc = 1.0 / (1.0 + np.exp(-diff))

    update = learning_rate * m_hat * dfc / (np.sqrt(v_hat) + epsilon)
    params_new = params - update

    return params_new, m_new, v_new, grads

def test_diffgrad():
    print("Testing diffGrad optimizer component...")
    np.random.seed(42)

    # Simple quadratic function: f(x) = x^2, optimal is x=0
    x = np.array([5.0])

    m = np.zeros_like(x)
    v = np.zeros_like(x)
    prev_g = np.zeros_like(x)

    learning_rate = 0.1
    epochs = 100

    for t in range(1, epochs + 1):
        # Forward pass & gradient
        grad = 2 * x
        loss = x**2

        # Optimizer step
        x, m, v, prev_g = diffgrad_optimizer(
            x, grad, m, v, prev_g, t, learning_rate=learning_rate
        )

        if t % 20 == 0:
            print(f"Epoch {t}: Loss = {loss[0]:.4f}, x = {x[0]:.4f}")

    assert loss < 5.0, "Optimizer did not converge effectively"
    print("diffGrad component tested successfully.")

if __name__ == "__main__":
    test_diffgrad()
