import numpy as np

def softsign(x):
    """
    Computes the Softsign activation function.
    f(x) = x / (1 + |x|)
    """
    return x / (1.0 + np.abs(x))

def softsign_derivative(x):
    """
    Computes the derivative of the Softsign activation function.
    f'(x) = 1 / (1 + |x|)^2
    """
    return 1.0 / (1.0 + np.abs(x))**2

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

class Layer:
    def __init__(self, input_size, output_size):
        # Initialize weights with standard normal distribution scaled for softsign
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = softsign(self.z)
        return self.a

    def backward(self, grad_output, learning_rate):
        grad_z = grad_output * softsign_derivative(self.z)
        grad_weights = np.dot(self.inputs.T, grad_z)
        grad_biases = np.sum(grad_z, axis=0, keepdims=True)
        grad_inputs = np.dot(grad_z, self.weights.T)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_inputs

def train_softsign_network():
    np.random.seed(42)
    # Simple XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    layer1 = Layer(2, 4)
    layer2 = Layer(4, 1)

    epochs = 10000
    learning_rate = 0.5

    for epoch in range(epochs):
        # Forward pass
        a1 = layer1.forward(X)
        a2 = layer2.forward(a1)

        # Compute loss
        loss = mse_loss(y, a2)

        # Backward pass
        grad_a2 = mse_loss_derivative(y, a2)
        grad_a1 = layer2.backward(grad_a2, learning_rate)
        layer1.backward(grad_a1, learning_rate)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Final Loss:", loss)
    print("Predictions:")
    print(a2)

if __name__ == "__main__":
    train_softsign_network()
