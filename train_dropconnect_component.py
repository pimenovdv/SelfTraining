import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class DropConnectLayer:
    def __init__(self, input_dim, output_dim, drop_prob=0.5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_prob = drop_prob
        # Initialize weights and bias
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros((1, output_dim))
        self.M = None
        self.x = None

    def forward(self, x, is_training=True):
        self.x = x
        if is_training:
            # Generate mask M with shape of W
            keep_prob = 1.0 - self.drop_prob
            self.M = (np.random.rand(*self.W.shape) < keep_prob) / keep_prob
            # Apply mask to W
            self.W_masked = self.W * self.M
        else:
            # During inference, use expected weights
            self.W_masked = self.W

        return np.dot(x, self.W_masked) + self.b

    def backward(self, dout, lr):
        # Gradients with respect to masked weights
        dW_masked = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)

        # Backpropagate error to input
        dx = np.dot(dout, self.W_masked.T)

        # Gradient with respect to actual weights W
        # The mask was applied during forward pass, so dW = dW_masked * M
        dW = dW_masked * self.M

        # Update weights and biases
        self.W -= lr * dW
        self.b -= lr * db

        return dx

def test_dropconnect():
    print("Testing DropConnect Component...")
    np.random.seed(42)

    # Simple dataset: XOR-like problem but mapped to larger dimensions
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    layer1 = DropConnectLayer(2, 16, drop_prob=0.2)
    layer2 = DropConnectLayer(16, 1, drop_prob=0.2)

    epochs = 10000
    lr = 0.5

    for epoch in range(epochs):
        # Forward pass
        z1 = layer1.forward(X, is_training=True)
        a1 = sigmoid(z1)
        z2 = layer2.forward(a1, is_training=True)
        a2 = sigmoid(z2)

        # Compute loss
        loss = np.mean((a2 - y) ** 2)

        # Backward pass
        da2 = 2 * (a2 - y) / y.size
        dz2 = da2 * sigmoid_derivative(z2)
        da1 = layer2.backward(dz2, lr)
        dz1 = da1 * sigmoid_derivative(z1)
        layer1.backward(dz1, lr)

        if epoch % 2000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Evaluation
    print("\nEvaluating DropConnect on Training Data:")
    z1_eval = layer1.forward(X, is_training=False)
    a1_eval = sigmoid(z1_eval)
    z2_eval = layer2.forward(a1_eval, is_training=False)
    predictions = sigmoid(z2_eval)

    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")

    final_loss = np.mean((predictions - y) ** 2)
    print(f"Final Evaluation Loss: {final_loss:.4f}")

if __name__ == "__main__":
    test_dropconnect()
