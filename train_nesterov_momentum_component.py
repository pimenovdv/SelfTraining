import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_nesterov_momentum_component():
    """
    Evaluates a Nesterov Accelerated Gradient (NAG) Optimizer mathematically.
    Uses a simple convex objective function f(x) = x^2 to verify that NAG
    can optimize it using lookahead momentum.
    """
    # Simple objective: f(x) = x^2
    # Gradient: f'(x) = 2x
    x = np.array([10.0]) # Initial point
    learning_rate = 0.1
    momentum = 0.9
    velocity = np.zeros_like(x)

    epochs = 50
    for epoch in range(epochs):
        # Nesterov lookahead: compute gradient at x + momentum * velocity
        lookahead_x = x + momentum * velocity
        grad_lookahead = 2 * lookahead_x

        # Update velocity
        velocity = momentum * velocity - learning_rate * grad_lookahead

        # Update position
        x = x + velocity

        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}, x: {x[0]:.4f}, f(x): {(x[0]**2):.4f}")

    final_loss = x[0]**2
    logging.info(f"Final x: {x[0]:.4f}, Final Loss: {final_loss:.4f}")
    assert final_loss < 1e-4, "Nesterov Momentum Optimizer failed to converge."
    return True

if __name__ == "__main__":
    train_nesterov_momentum_component()
