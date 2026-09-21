import numpy as np

def log_cosh_loss(y_true, y_pred):
    """
    Computes the Log-Cosh loss.
    L(y, y') = sum(log(cosh(y' - y)))
    """
    x = y_pred - y_true
    # log(cosh(x)) = x + softplus(-2x) - log(2) for better numerical stability
    # Or simply: np.log(np.cosh(x))
    # We use np.log(np.cosh(x)) and clip to prevent overflow.
    return np.mean(np.log(np.cosh(x + 1e-12)))

def log_cosh_loss_derivative(y_true, y_pred):
    """
    Computes the derivative of the Log-Cosh loss with respect to y_pred.
    dL/dy' = tanh(y' - y)
    """
    x = y_pred - y_true
    return np.tanh(x) / y_true.size

class LinearLayer:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        # Initialize weights and biases
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.bias = np.zeros((1, output_dim))
        self.learning_rate = learning_rate

    def forward(self, X):
        self.input = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, d_output):
        # d_output: gradient of loss w.r.t. output
        d_weights = np.dot(self.input.T, d_output)
        d_bias = np.sum(d_output, axis=0, keepdims=True)
        d_input = np.dot(d_output, self.weights.T)

        # Update weights and biases
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias

        return d_input

def test_log_cosh_loss_component():
    np.random.seed(42)

    # Linear regression dataset
    X = np.random.randn(100, 3)
    true_weights = np.array([[1.5], [-2.0], [0.5]])
    y_true = np.dot(X, true_weights) + 0.1 * np.random.randn(100, 1)

    model = LinearLayer(input_dim=3, output_dim=1, learning_rate=0.1)

    epochs = 100
    loss_history = []

    for epoch in range(epochs):
        # Forward pass
        y_pred = model.forward(X)

        # Compute loss
        loss = log_cosh_loss(y_true, y_pred)
        loss_history.append(loss)

        # Compute loss derivative
        d_loss = log_cosh_loss_derivative(y_true, y_pred)

        # Backward pass
        model.backward(d_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Initial loss:", loss_history[0])
    print("Final loss:", loss_history[-1])

    assert loss_history[-1] < loss_history[0], "Loss did not decrease over time."
    print("Log-Cosh Loss Component tested successfully.")

if __name__ == "__main__":
    test_log_cosh_loss_component()
