import numpy as np

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def train_elu():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    # Initialize weights
    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))

    alpha = 1.0
    lr = 0.5
    epochs = 1000

    print("Training ELU Network...")
    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = elu(z1, alpha)
        z2 = np.dot(a1, W2) + b2
        # Use sigmoid for the output layer
        a2 = 1 / (1 + np.exp(-z2))

        # Loss (Binary Cross-Entropy)
        loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))

        # Backward pass
        dz2 = a2 - y
        dW2 = np.dot(a1.T, dz2) / X.shape[0]
        db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * elu_derivative(z1, alpha)
        dW1 = np.dot(X.T, dz1) / X.shape[0]
        db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]

        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")

    print("Training Complete.")
    print("Predictions:")
    print(np.round(a2))

if __name__ == '__main__':
    train_elu()
