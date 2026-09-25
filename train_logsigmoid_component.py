import numpy as np

class LogSigmoid:
    """
    LogSigmoid Activation Function.
    Calculates log(1 / (1 + exp(-x))) = log(sigmoid(x)).
    Often used in combination with loss functions or when probability distributions
    need to be analyzed in log-space for numerical stability.
    """
    def __init__(self):
        self.inputs = None

    def forward(self, x):
        """
        Forward pass of the LogSigmoid activation.
        Uses -np.logaddexp(0, -x) for numerical stability.
        """
        self.inputs = x
        return -np.logaddexp(0, -x)

    def backward(self, grad_output):
        """
        Backward pass for LogSigmoid.
        Derivative of LogSigmoid is 1 - sigmoid(x).
        """
        x = self.inputs
        sigmoid = np.where(x >= 0,
                           1 / (1 + np.exp(-x)),
                           np.exp(x) / (1 + np.exp(x)))
        grad_input = grad_output * (1 - sigmoid)
        return grad_input

def test_logsigmoid_component():
    np.random.seed(42)
    # Define an artificial dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Targets for a simple logical OR task to show that LogSigmoid can learn bounds
    y = np.array([[0], [1], [1], [1]])

    # Define simple model parameters
    input_size = 2
    hidden_size = 4
    output_size = 1

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((1, output_size))

    activation = LogSigmoid()

    learning_rate = 0.1
    epochs = 2000

    print("Training a simple Multi-Layer Perceptron using LogSigmoid activation on the OR dataset...")
    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = activation.forward(z1)
        z2 = np.dot(a1, W2) + b2
        # Using sigmoid for final output to match probabilities [0, 1]
        a2 = 1 / (1 + np.exp(-z2))

        # Calculate loss (MSE)
        loss = np.mean((a2 - y) ** 2)

        # Backward pass
        # Gradient of MSE and Sigmoid
        dz2 = (a2 - y) * a2 * (1 - a2)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = activation.backward(da1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    print("\nFinal Predictions vs Targets:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {a2[i][0]:.4f}")

    # Basic verification
    assert loss < 0.1, f"Loss too high: {loss}. LogSigmoid should help model converge."
    print("LogSigmoid component test completed successfully!")

if __name__ == "__main__":
    test_logsigmoid_component()
