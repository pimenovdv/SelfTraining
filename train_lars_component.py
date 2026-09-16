import numpy as np

def lars_optimizer_step(weights, gradients, velocities, lr, momentum, weight_decay, eta=0.001):
    """
    Performs a single step of Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    """
    w_norm = np.linalg.norm(weights)
    g_norm = np.linalg.norm(gradients)

    if w_norm > 0 and g_norm > 0:
        local_lr = eta * w_norm / (g_norm + weight_decay * w_norm + 1e-8)
    else:
        local_lr = 1.0

    actual_lr = lr * local_lr

    # Weight decay is applied to gradients before momentum
    if weight_decay > 0:
        gradients = gradients + weight_decay * weights

    velocities = momentum * velocities + actual_lr * gradients
    weights = weights - velocities

    return weights, velocities

def test_lars_optimizer():
    print("Testing LARS Optimizer Component...")
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_w = np.array([1.5, -2.0, 3.0, -0.5, 2.2])
    y = X.dot(true_w) + np.random.randn(100) * 0.1

    # Initialize weights to non-zero so w_norm > 0 initially
    weights = np.ones(5) * 0.1
    velocities = np.zeros(5)

    # With LARS, actual learning rate is scaled down heavily, so base LR can be larger
    lr = 10.0
    momentum = 0.9
    weight_decay = 1e-4

    epochs = 500
    for epoch in range(epochs):
        predictions = X.dot(weights)
        errors = predictions - y
        gradients = (2.0 / len(X)) * X.T.dot(errors)

        weights, velocities = lars_optimizer_step(
            weights, gradients, velocities, lr, momentum, weight_decay, eta=0.01
        )

        if epoch % 100 == 0:
            loss = np.mean(errors**2)
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    final_loss = np.mean((X.dot(weights) - y)**2)
    print(f"Final Loss = {final_loss:.4f}")
    assert final_loss < 0.2, "Loss did not converge"
    print("LARS optimizer component successfully trained!")

if __name__ == "__main__":
    test_lars_optimizer()
