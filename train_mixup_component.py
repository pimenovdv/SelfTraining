import numpy as np

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def bce_loss(pred, target):
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

def bce_gradient(pred, target):
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return (pred - target) / (pred * (1 - pred)) / target.shape[0]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_mixup():
    print("Testing Mixup Component mathematically in pure NumPy...")

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))

    learning_rate = 0.5
    epochs = 2000

    for epoch in range(epochs):
        mixed_X, y_a, y_b, lam = mixup_data(X, y, alpha=0.5)

        # Forward pass
        z1 = np.dot(mixed_X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        loss = mixup_criterion(bce_loss, a2, y_a, y_b, lam)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # Backward pass
        # The gradient of the mixup loss w.r.t a2 is a linear combination of the BCE gradients
        grad_a2 = lam * bce_gradient(a2, y_a) + (1 - lam) * bce_gradient(a2, y_b)

        grad_z2 = grad_a2 * sigmoid_derivative(z2)
        grad_W2 = np.dot(a1.T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)

        grad_a1 = np.dot(grad_z2, W2.T)
        grad_z1 = grad_a1 * sigmoid_derivative(z1)
        grad_W1 = np.dot(mixed_X.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1

    # Test the model on original data
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    predictions = sigmoid(z2)

    print("\nPredictions on original data:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")

    mse = np.mean((predictions - y) ** 2)
    print(f"Final MSE: {mse:.4f}")

    if mse < 0.1:
        print("Mixup component training successful.")
        return True
    else:
        print("Mixup component failed to converge.")
        return False

if __name__ == "__main__":
    train_mixup()
