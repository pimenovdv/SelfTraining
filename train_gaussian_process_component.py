import numpy as np

def rbf_kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Computes the RBF kernel between X1 and X2.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def main():
    print("Initializing Gaussian Process Regression component test...")
    np.random.seed(42)

    # Training data
    X_train = np.random.uniform(-5, 5, 15).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, X_train.shape)

    # Test data
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)

    # GP Hyperparameters
    l = 1.0
    sigma_f = 1.0
    sigma_y = 0.1

    # Compute kernels
    K = rbf_kernel(X_train, X_train, l, sigma_f)
    K_s = rbf_kernel(X_train, X_test, l, sigma_f)
    K_ss = rbf_kernel(X_test, X_test, l, sigma_f)

    # Cholesky decomposition for numerical stability
    L = np.linalg.cholesky(K + sigma_y**2 * np.eye(len(X_train)))

    # Predictive mean
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu_s = np.dot(K_s.T, alpha)

    # Predictive variance
    v = np.linalg.solve(L, K_s)
    cov_s = K_ss - np.dot(v.T, v)
    std_s = np.sqrt(np.diag(cov_s))

    # Evaluate Log Marginal Likelihood
    lml = -0.5 * np.dot(y_train.T, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * len(X_train) * np.log(2 * np.pi)
    lml = lml[0, 0]

    # Calculate MSE on train
    y_pred_train = np.dot(K.T, alpha)
    train_mse = np.mean((y_train - y_pred_train)**2)

    # Output results
    print(f"Log Marginal Likelihood: {lml:.4f}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Mean test std (uncertainty): {np.mean(std_s):.4f}")

    # Verify predictions are reasonable
    assert train_mse < 0.1, f"Training MSE too high: {train_mse}"
    print("Gaussian Process Regression test passed successfully.")

if __name__ == "__main__":
    main()
