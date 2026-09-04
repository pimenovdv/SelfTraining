import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_perceptron(X, y, epochs=1000, lr=0.1):
    np.random.seed(42)
    weights = np.random.randn(X.shape[1])
    bias = np.random.randn(1)

    for epoch in range(epochs):
        # Forward pass
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)

        # Loss (Binary Cross Entropy)
        loss = -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))

        # Backward pass
        dz = predictions - y
        dw = np.dot(X.T, dz) / len(y)
        db = np.sum(dz) / len(y)

        # Update weights
        weights -= lr * dw
        bias -= lr * db

    return weights, bias

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1]) # AND gate

    weights, bias = train_perceptron(X, y, epochs=10000, lr=0.1)
    print("Trained weights:", weights)
    print("Trained bias:", bias)

    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)
    print("Predictions:", predictions)
    assert np.all((predictions > 0.5) == y)
    print("Perceptron training successful!")
