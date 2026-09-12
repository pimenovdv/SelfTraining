import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros(self.W1.shape[1])
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros(self.b1.shape[1])

        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros(self.W2.shape[1])
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros(self.b2.shape[1])

        self.t = 0

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8, weight_decay=0.0):
        m = X.shape[0]

        d_a2 = mse_derivative(y, self.a2)
        d_z2 = d_a2 * sigmoid_derivative(self.z2)
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * sigmoid_derivative(self.z1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        self.t += 1

        def novograd_update(param, grad, m, v):
            g_norm = np.sum(grad ** 2, axis=0)

            if self.t == 1:
                v = g_norm
            else:
                v = beta2 * v + (1 - beta2) * g_norm

            grad_scaled = grad / (np.sqrt(v + epsilon))

            m = beta1 * m + (1 - beta1) * (grad_scaled + weight_decay * param)

            param -= learning_rate * m
            return param, m, v

        self.W1, self.m_W1, self.v_W1 = novograd_update(self.W1, d_W1, self.m_W1, self.v_W1)
        self.b1, self.m_b1, self.v_b1 = novograd_update(self.b1, d_b1, self.m_b1, self.v_b1)
        self.W2, self.m_W2, self.v_W2 = novograd_update(self.W2, d_W2, self.m_W2, self.v_W2)
        self.b2, self.m_b2, self.v_b2 = novograd_update(self.b2, d_b2, self.m_b2, self.v_b2)

def test_component():
    print("Testing NovoGrad Optimizer Component...")

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    epochs = 5000
    learning_rate = 0.05

    for epoch in range(epochs):
        predictions = nn.forward(X)
        loss = mse_loss(y, predictions)
        nn.backward(X, y, learning_rate)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    final_predictions = nn.forward(X)
    print("Final Predictions:")
    print(final_predictions)

    predictions_binary = (final_predictions > 0.5).astype(int)
    accuracy = np.mean(predictions_binary == y)
    print(f"Accuracy: {accuracy * 100}%")

    if accuracy == 1.0:
        print("Success: The network learned the XOR mapping using NovoGrad optimizer.")
    else:
        print("Failure: The network failed to learn the XOR mapping.")

if __name__ == "__main__":
    test_component()
