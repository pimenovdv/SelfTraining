import numpy as np

def ftrl_update(params, z, n, grads, alpha, beta, l1, l2):
    """
    Performs a single Follow The Regularized Leader (FTRL-Proximal) update.
    """
    updated_params = []

    for i in range(len(params)):
        # Accumulate squared gradients
        g = grads[i]

        # Calculate sigma: the learning rate adjustment
        sigma = (np.sqrt(n[i] + g**2) - np.sqrt(n[i])) / alpha

        # Accumulate gradients (z)
        z[i] += g - sigma * params[i]

        # Update squared gradients sum
        n[i] += g**2

        # Determine signs
        sign_z = np.sign(z[i])

        # FTRL logic: soft thresholding
        mask = np.abs(z[i]) > l1

        # Compute weights where the condition is met, otherwise 0
        w = np.zeros_like(params[i])

        # Compute new weight based on FTRL formula for active elements
        # Since active is a boolean mask array, we can use np.where
        term1 = sign_z * l1 - z[i]
        term2 = (beta + np.sqrt(n[i])) / alpha + l2

        w_active = term1 / term2

        # Apply mask
        w = np.where(mask, w_active, 0.0)

        updated_params.append(w)

    return updated_params, z, n

def test_ftrl():
    # Simple logistic regression dataset for binary classification
    np.random.seed(42)

    # Generate data: 1000 samples, 5 features
    X = np.random.randn(1000, 5)

    # Make features 1, 3, 4 completely irrelevant (just noise)
    # Make features 0 and 2 highly predictive
    true_w = np.array([[3.0], [0.0], [-2.5], [0.0], [0.0]])
    logits = X.dot(true_w)

    # Sigmoid
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(float)

    # Initialize parameters
    w = np.zeros((5, 1))
    params = [w]

    # FTRL Hyperparameters
    alpha = 0.5 # Higher Learning rate
    beta = 1.0
    l1 = 150.0 # High L1 regularization strength to induce sparsity
    l2 = 1.0 # L2 regularization strength

    # State variables for FTRL
    z = [np.zeros_like(p) for p in params]
    n = [np.zeros_like(p) for p in params]

    epochs = 100

    for epoch in range(epochs):
        # Forward pass (Sigmoid)
        logits = X.dot(params[0])
        preds = 1 / (1 + np.exp(-logits))

        # Loss (Binary Cross-Entropy)
        # Add epsilon to prevent log(0)
        eps = 1e-15
        preds = np.clip(preds, eps, 1 - eps)
        loss = -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))

        # Gradients
        # For logistic regression: sum over batch of X^T (pred - y)
        grad_w = X.T.dot(preds - y)
        grads = [grad_w]

        # Update parameters using FTRL
        params, z, n = ftrl_update(params, z, n, grads, alpha, beta, l1, l2)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Final parameters:\n", params[0])
    print("True parameters:\n", true_w)

    # Verify that irrelevant features were driven exactly to zero by L1 regularization in FTRL
    assert params[0][1] == 0.0, f"Feature 1 should be sparse (0.0), got {params[0][1]}"
    assert params[0][3] == 0.0, f"Feature 3 should be sparse (0.0), got {params[0][3]}"
    assert params[0][4] == 0.0, f"Feature 4 should be sparse (0.0), got {params[0][4]}"

    # Verify relevant features moved in the right direction
    assert params[0][0] > 0.5, "Feature 0 should be positive"
    assert params[0][2] < -0.5, "Feature 2 should be negative"

    print("FTRL component test passed! Sparsity achieved.")

if __name__ == "__main__":
    test_ftrl()
