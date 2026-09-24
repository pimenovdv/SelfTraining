import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_backward(x, grad_output):
    return grad_output * (x > 0)

# XOR data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)
# ReGLU uses two projections, so we project to 2 * hidden_dim
# Let's say hidden_dim = 4. Total projected dim = 8.
W1 = np.random.randn(2, 4) * 0.1
b1 = np.zeros((1, 4))
W2 = np.random.randn(2, 4) * 0.1
b2 = np.zeros((1, 4))

# Output layer
W3 = np.random.randn(4, 1) * 0.1
b3 = np.zeros((1, 1))

lr = 0.5
epochs = 5000

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    z2 = np.dot(X, W2) + b2

    a1 = relu(z1)
    # ReGLU activation
    reglu_out = a1 * z2

    z3 = np.dot(reglu_out, W3) + b3
    a3 = 1 / (1 + np.exp(-np.clip(z3, -500, 500)))

    # Loss (Binary Cross Entropy)
    loss = -np.mean(y * np.log(a3 + 1e-15) + (1 - y) * np.log(1 - a3 + 1e-15))

    # Backward pass
    dz3 = a3 - y
    dW3 = np.dot(reglu_out.T, dz3) / len(X)
    db3 = np.sum(dz3, axis=0, keepdims=True) / len(X)

    dreglu_out = np.dot(dz3, W3.T)

    # reglu_out = a1 * z2
    # d(reglu_out)/d(z2) = a1
    # d(reglu_out)/d(a1) = z2

    da1 = dreglu_out * z2
    dz2 = dreglu_out * a1

    dz1 = relu_backward(z1, da1)

    dW2 = np.dot(X.T, dz2) / len(X)
    db2 = np.sum(dz2, axis=0, keepdims=True) / len(X)

    dW1 = np.dot(X.T, dz1) / len(X)
    db1 = np.sum(dz1, axis=0, keepdims=True) / len(X)

    # Update weights
    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

print(f"Final Loss: {loss:.4f}")
predictions = (a3 > 0.5).astype(int)
print("Predictions:\n", predictions)

assert loss < 0.1, "Loss should converge to less than 0.1 on XOR"
print("ReGLU component trained successfully.")
