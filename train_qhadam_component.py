import numpy as np

def qhadam_step(params, grads, exp_avg, exp_avg_sq, v2, lr, beta1, beta2, nus, eps, step):
    """
    Quasi-Hyperbolic Adam (QHAdam) optimization step.

    Args:
        params: Array of parameters.
        grads: Array of gradients.
        exp_avg: Exponential moving average of gradient values.
        exp_avg_sq: Exponential moving average of squared gradient values.
        v2: Moving average of squared gradients without discount in step.
        lr: Learning rate.
        beta1: Exponential decay rate for first moment estimates.
        beta2: Exponential decay rate for second moment estimates.
        nus: Tuple of discount factors (nu1, nu2).
        eps: Term added to the denominator to improve numerical stability.
        step: Current step number.

    Returns:
        Updated params, exp_avg, exp_avg_sq
    """
    nu1, nu2 = nus

    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1 - beta1) * grads

    # Update biased second raw moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grads ** 2)

    # Bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    # Compute the modified first moment (QH momentum)
    m_hat = ( (1 - nu1) * grads + nu1 * exp_avg ) / bias_correction1

    # Compute the modified second moment
    v_hat = ( (1 - nu2) * (grads ** 2) + nu2 * exp_avg_sq ) / bias_correction2

    # Update parameters
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

    return params, exp_avg, exp_avg_sq

def train_qhadam():
    """
    Trains a simple linear model using QHAdam optimizer on a toy dataset.
    """
    # Toy dataset: y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([[3.0], [5.0], [7.0], [9.0]])

    # Initialize parameters: weights and bias
    W = np.random.randn(1, 1)
    b = np.random.randn(1)

    # Optimizer hyperparameters
    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    nus = (0.7, 1.0) # (nu1, nu2)
    eps = 1e-8
    epochs = 200

    # Initialize momentum states
    exp_avg_W = np.zeros_like(W)
    exp_avg_sq_W = np.zeros_like(W)

    exp_avg_b = np.zeros_like(b)
    exp_avg_sq_b = np.zeros_like(b)

    for epoch in range(1, epochs + 1):
        # Forward pass
        y_pred = X.dot(W) + b

        # Compute loss (MSE)
        loss = np.mean((y_pred - y) ** 2)

        # Compute gradients
        grad_y_pred = 2 * (y_pred - y) / X.shape[0]
        grad_W = X.T.dot(grad_y_pred)
        grad_b = np.sum(grad_y_pred, axis=0)

        # Update parameters using QHAdam
        W, exp_avg_W, exp_avg_sq_W = qhadam_step(
            W, grad_W, exp_avg_W, exp_avg_sq_W, None, lr, beta1, beta2, nus, eps, epoch
        )
        b, exp_avg_b, exp_avg_sq_b = qhadam_step(
            b, grad_b, exp_avg_b, exp_avg_sq_b, None, lr, beta1, beta2, nus, eps, epoch
        )

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, W: {W[0,0]:.4f}, b: {b[0]:.4f}")

    return W, b

if __name__ == "__main__":
    train_qhadam()
