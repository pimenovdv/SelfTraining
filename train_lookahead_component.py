import numpy as np

def train_lookahead_component():
    """
    Evaluates a Lookahead Optimizer component mathematically in pure NumPy.
    """
    print("Initializing Lookahead Optimizer Component...")

    # Generate synthetic data for linear regression
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_w = np.array([1.5, -2.0, 0.5, 0.0, -1.0])
    y = X.dot(true_w) + np.random.randn(100) * 0.1

    # Initialize weights
    w_fast = np.zeros(5)
    w_slow = np.zeros(5)

    # Hyperparameters
    lr = 0.05
    alpha = 0.5 # Slow weight step size
    k = 5 # Lookahead steps
    epochs = 100

    print(f"Initial weights: {w_fast}")

    for epoch in range(epochs):
        for step in range(k):
            # Forward pass
            predictions = X.dot(w_fast)

            # Compute loss (MSE)
            loss = np.mean((predictions - y) ** 2)

            # Backward pass (gradient)
            grad = 2 * X.T.dot(predictions - y) / len(y)

            # Fast update (SGD)
            w_fast -= lr * grad

        # Slow update (Lookahead)
        w_slow = w_slow + alpha * (w_fast - w_slow)
        w_fast = np.copy(w_slow)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Final evaluation
    final_loss = np.mean((X.dot(w_fast) - y) ** 2)
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Weights: {w_fast}")
    print(f"True Weights: {true_w}")

    assert final_loss < 0.1, "Lookahead Optimizer failed to converge."
    print("Lookahead Optimizer Component successfully converged!")

if __name__ == "__main__":
    train_lookahead_component()
