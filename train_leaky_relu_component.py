import numpy as np

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def leaky_relu_backward(x, grad_output, alpha=0.01):
    grad = np.where(x > 0, 1.0, alpha)
    return grad_output * grad

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# XOR data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)
W1 = np.random.randn(2, 4) * 1.0
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 1.0
b2 = np.zeros((1, 1))

lr = 0.5
epochs = 5000

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Loss (Binary Cross Entropy)
    loss = -np.mean(y * np.log(a2 + 1e-15) + (1 - y) * np.log(1 - a2 + 1e-15))

    # Backward pass
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / len(X)
    db2 = np.sum(dz2, axis=0, keepdims=True) / len(X)

    da1 = np.dot(dz2, W2.T)
    dz1 = leaky_relu_backward(z1, da1)

    dW1 = np.dot(X.T, dz1) / len(X)
    db1 = np.sum(dz1, axis=0, keepdims=True) / len(X)

    # Update weights
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

print(f"Final Loss: {loss:.4f}")
predictions = (a2 > 0.5).astype(int)
print("Predictions:\n", predictions)

assert loss < 0.1, "Loss should converge to less than 0.1 on XOR"
print("Leaky ReLU component trained successfully.")
