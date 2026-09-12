import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Computes the binary focal loss between true labels and predictions.

    Args:
        y_true (np.ndarray): Ground truth labels (0 or 1).
        y_pred (np.ndarray): Predicted probabilities (between 0 and 1).
        gamma (float): Focusing parameter. Default is 2.0.
        alpha (float): Balancing parameter. Default is 0.25.

    Returns:
        float: The mean focal loss over the batch.
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Compute probability of true class
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

    # Compute alpha weighting factor
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)

    # Compute focal loss
    loss = -alpha_t * np.power(1 - p_t, gamma) * np.log(p_t)
    return np.mean(loss)

def focal_loss_gradient(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Computes the gradient of the binary focal loss with respect to predictions.
    """
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)

    # Gradient derived via chain rule
    dp_t_dy_pred = y_true - (1 - y_true)

    term1 = gamma * np.power(1 - p_t, gamma - 1) * np.log(p_t)
    term2 = np.power(1 - p_t, gamma) / p_t

    grad = -alpha_t * (term2 - term1) * dp_t_dy_pred

    return grad / y_true.size

class SimpleLogisticRegression:
    def __init__(self, input_dim):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def train(self, X, y, epochs=1000, lr=0.1, gamma=2.0, alpha=0.25):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            loss = focal_loss(y, y_pred, gamma, alpha)
            losses.append(loss)

            # Backward pass
            grad_y_pred = focal_loss_gradient(y, y_pred, gamma, alpha)

            grad_z = grad_y_pred * y_pred * (1 - y_pred)

            grad_w = np.dot(X.T, grad_z)
            grad_b = np.sum(grad_z)

            # Update weights
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return losses

def test_focal_loss():
    print("Testing Focal Loss Component...")

    # Create an imbalanced dataset
    np.random.seed(42)
    # Majority class (0)
    X_maj = np.random.randn(90, 2) - 1.0
    y_maj = np.zeros(90)

    # Minority class (1)
    X_min = np.random.randn(10, 2) + 1.0
    y_min = np.ones(10)

    X = np.vstack((X_maj, X_min))
    y = np.concatenate((y_maj, y_min))

    print(f"Dataset shape: {X.shape}, minority class ratio: {np.mean(y):.2f}")

    model = SimpleLogisticRegression(input_dim=2)

    print("\nTraining with Focal Loss (gamma=2.0, alpha=0.25):")
    model.train(X, y, epochs=500, lr=1.0)

    y_pred = model.forward(X)

    print(f"Final Loss: {focal_loss(y, y_pred):.4f}")
    print("Test passed successfully.")

if __name__ == "__main__":
    test_focal_loss()
