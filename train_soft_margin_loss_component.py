import numpy as np

def soft_margin_loss(predictions, targets):
    """
    Computes the Soft Margin Loss.
    targets should be in {-1, 1}.
    """
    return np.mean(np.log1p(np.exp(-targets * predictions)))

def soft_margin_loss_gradient(predictions, targets):
    """
    Computes the gradient of Soft Margin Loss with respect to predictions.
    """
    exp_term = np.exp(-targets * predictions)
    return (-targets * exp_term) / (1 + exp_term) / targets.size

def test_soft_margin_loss():
    print("Initializing Soft Margin Loss component test...")
    np.random.seed(42)

    # Generate some toy data for binary classification (labels -1 and 1)
    X = np.random.randn(100, 2)
    true_w = np.array([1.5, -2.0])
    y = np.sign(X.dot(true_w) + 0.5 * np.random.randn(100))
    y[y == 0] = 1 # Avoid 0 labels

    # Initialize weights
    w = np.zeros(2)

    learning_rate = 0.1
    epochs = 100

    print("Starting training...")
    for epoch in range(epochs):
        predictions = X.dot(w)
        loss = soft_margin_loss(predictions, y)

        grad_pred = soft_margin_loss_gradient(predictions, y)
        grad_w = X.T.dot(grad_pred)

        w -= learning_rate * grad_w

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    final_predictions = X.dot(w)
    final_loss = soft_margin_loss(final_predictions, y)
    print("Training complete.")
    print(f"Final Weights: {w}")
    print(f"Final Loss: {final_loss:.4f}")

    # Check if loss has decreased
    initial_loss = np.log(2) # loss when w = 0 is log(1 + exp(0)) = log(2) = 0.693
    success = final_loss < initial_loss
    if success:
        print("Success: Soft Margin Loss successfully trained the model.")
    else:
        print("Failure: Loss did not decrease.")

if __name__ == '__main__':
    test_soft_margin_loss()
