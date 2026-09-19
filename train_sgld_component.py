import numpy as np

def sgld_step(params, grads, lr):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) update step.

    Args:
        params (list of np.ndarray): Current parameters.
        grads (list of np.ndarray): Current gradients.
        lr (float): Learning rate.

    Returns:
        list of np.ndarray: Updated parameters.
    """
    updated_params = []
    for p, g in zip(params, grads):
        noise = np.random.normal(0, np.sqrt(2 * lr), size=p.shape)
        p_new = p - lr * g + noise
        updated_params.append(p_new)
    return updated_params

def test_sgld_component():
    print("Testing Stochastic Gradient Langevin Dynamics (SGLD) component...")
    np.random.seed(42)

    params = [np.array([5.0])]
    lr = 0.01

    print(f"Initial param: {params[0][0]:.4f}")

    for i in range(1000):
        grads = [2 * params[0]]
        params = sgld_step(params, grads, lr)

    print(f"Final param after 1000 steps: {params[0][0]:.4f}")

    # SGLD doesn't converge to exactly 0, it samples around 0 with variance related to lr
    assert np.abs(params[0][0]) < 2.0, "SGLD should sample near the minimum"
    print("SGLD component test passed!")

if __name__ == "__main__":
    test_sgld_component()
