import numpy as np

def orthogonal_matching_pursuit(X, y, n_nonzero_coefs):
    """
    Orthogonal Matching Pursuit (OMP) algorithm.
    X: Dictionary matrix of shape (n_samples, n_features)
    y: Target vector of shape (n_samples,)
    n_nonzero_coefs: Number of non-zero coefficients to keep
    """
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    residual = y.copy()
    selected_indices = []

    for _ in range(n_nonzero_coefs):
        # Compute correlations
        correlations = np.abs(X.T @ residual)

        # Avoid re-selecting the same feature
        correlations[selected_indices] = 0

        # Select the best feature
        best_idx = np.argmax(correlations)
        selected_indices.append(best_idx)

        # Orthogonal projection using selected features
        X_selected = X[:, selected_indices]
        coef_selected, _, _, _ = np.linalg.lstsq(X_selected, y, rcond=None)

        # Update residual
        residual = y - X_selected @ coef_selected

        # Update coefficients
        coef[selected_indices] = coef_selected

    return coef

def test_omp():
    np.random.seed(42)
    n_samples, n_features = 50, 100
    n_nonzero_coefs = 5

    # Generate random dictionary
    X = np.random.randn(n_samples, n_features)
    # Normalize dictionary columns
    X /= np.linalg.norm(X, axis=0)

    # Generate sparse coefficients
    true_coef = np.zeros(n_features)
    true_indices = np.random.choice(n_features, n_nonzero_coefs, replace=False)
    true_coef[true_indices] = np.random.randn(n_nonzero_coefs)

    # Generate target
    y = X @ true_coef

    # Add some noise
    y += 0.01 * np.random.randn(n_samples)

    # Run OMP
    estimated_coef = orthogonal_matching_pursuit(X, y, n_nonzero_coefs)

    # Evaluate
    mse = np.mean((X @ estimated_coef - y) ** 2)
    print(f"MSE: {mse}")

    # Check if selected indices match
    estimated_indices = np.nonzero(estimated_coef)[0]
    overlap = len(set(true_indices).intersection(set(estimated_indices)))
    print(f"Recovered {overlap}/{n_nonzero_coefs} non-zero coefficients.")

    assert mse < 0.1, "MSE is too high, OMP failed."
    print("OMP component tested successfully.")

if __name__ == "__main__":
    test_omp()
