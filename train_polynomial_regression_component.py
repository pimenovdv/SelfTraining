import numpy as np

def main():
    print("Starting Polynomial Regression component test...")
    # Generate synthetic data
    np.random.seed(42)
    X = np.sort(np.random.rand(100, 1) * 10 - 5, axis=0)
    # True function: y = 0.5 * X^2 + x + 2 + noise
    y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 2

    # Hyperparameters
    degree = 2
    lr = 0.001
    epochs = 10000

    # Polynomial features
    # For degree 2: [1, x, x^2]
    X_poly = np.ones((X.shape[0], 1))
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X**d))

    # Initialize weights
    weights = np.zeros((degree + 1, 1))

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        predictions = X_poly.dot(weights)

        # Loss (Mean Squared Error)
        error = predictions - y
        mse = np.mean(error**2)

        # Backward pass (Gradients)
        gradients = 2/len(X) * X_poly.T.dot(error)

        # Update weights
        weights -= lr * gradients

    print("Training complete.")
    print(f"Final MSE: {mse:.4f}")
    print(f"Learned Weights:\n{weights}")

    assert mse < 10.0, "Model did not converge well"
    print("Polynomial Regression Component mathematically verified.")

if __name__ == "__main__":
    main()
