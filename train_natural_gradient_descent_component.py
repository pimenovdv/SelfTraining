"""
Natural Gradient Descent Component

This script implements Natural Gradient Descent (NGD) mathematically.
Unlike standard Gradient Descent which follows the steepest descent in Euclidean space,
NGD follows the steepest descent in the Riemannian manifold of probability distributions,
using the Fisher Information Matrix (FIM) as a local metric.

This allows NGD to be invariant to parameterization, often converging in fewer iterations.

For a linear regression model with Gaussian noise (MSE loss), the FIM is approximated
by the covariance of the inputs (E[X^T X]).
"""

import numpy as np

def natural_gradient_descent(X, y, learning_rate=0.1, epochs=100):
    """
    Performs Natural Gradient Descent for linear regression.

    Args:
        X (np.ndarray): Input features of shape (n_samples, n_features).
        y (np.ndarray): Target values of shape (n_samples,).
        learning_rate (float): The learning rate/step size.
        epochs (int): Number of training iterations.

    Returns:
        np.ndarray: Learned weights.
        list: History of loss values.
    """
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features)

    losses = []

    # Empirical Fisher Information Matrix (for Gaussian observation model)
    # Added small ridge for numerical stability (regularization)
    F = (X.T @ X) / n_samples + 1e-5 * np.eye(n_features)

    for epoch in range(epochs):
        # Linear predictions
        predictions = X @ weights

        # Mean Squared Error Loss
        error = predictions - y
        loss = 0.5 * np.mean(error**2)
        losses.append(loss)

        # Standard Euclidean gradient
        grad = X.T @ error / n_samples

        # Natural gradient direction: inverse(F) @ grad
        # Using np.linalg.solve is more stable than computing inverse explicitly
        natural_grad = np.linalg.solve(F, grad)

        # Update weights
        weights -= learning_rate * natural_grad

    return weights, losses

if __name__ == "__main__":
    print("Initializing Natural Gradient Descent Component...")
    np.random.seed(42)

    # Generate synthetic data
    n_samples = 200
    n_features = 4

    # Features with different scales to demonstrate NGD's robustness
    X = np.random.randn(n_samples, n_features) * np.array([1.0, 10.0, 0.1, 5.0])

    true_weights = np.array([1.5, -2.0, 0.5, 3.14])
    noise = np.random.randn(n_samples) * 0.5
    y = X @ true_weights + noise

    print(f"Data shape: {X.shape}, Targets shape: {y.shape}")
    print(f"True weights: {true_weights}")

    # Train using NGD
    print("\nTraining with Natural Gradient Descent...")
    weights_ngd, losses_ngd = natural_gradient_descent(X, y, learning_rate=0.1, epochs=50)

    print(f"\nFinal Learned Weights (NGD): {weights_ngd}")
    print(f"Final Loss (NGD): {losses_ngd[-1]:.4f}")

    # Compare with standard Gradient Descent
    print("\nTraining with Standard Gradient Descent for comparison...")
    weights_gd = np.random.RandomState(42).randn(n_features)
    losses_gd = []
    for _ in range(50):
        preds = X @ weights_gd
        err = preds - y
        losses_gd.append(0.5 * np.mean(err**2))
        grad = X.T @ err / n_samples
        # Standard GD needs a smaller learning rate here due to ill-conditioned X
        weights_gd -= 0.005 * grad

    print(f"Final Learned Weights (Standard GD): {weights_gd}")
    print(f"Final Loss (Standard GD): {losses_gd[-1]:.4f}")

    print("\nObservation: NGD converges much faster and is invariant to feature scaling,")
    print("while standard GD struggles with the ill-conditioned input matrix.")
    print("Component implemented successfully.")
