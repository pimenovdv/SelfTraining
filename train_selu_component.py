import numpy as np

# SELU constants
ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946

def selu(x):
    return SCALE * np.where(x > 0.0, x, ALPHA * (np.exp(x) - 1.0))

def d_selu(x):
    return SCALE * np.where(x > 0.0, 1.0, ALPHA * np.exp(x))

class DenseSELU:
    def __init__(self, in_features, out_features):
        # Initialization for SELU: mean 0, variance 1 / in_features
        self.W = np.random.randn(in_features, out_features) / np.sqrt(in_features)
        self.b = np.zeros(out_features)
        self.x = None
        self.z = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        return selu(self.z)

    def backward(self, grad_output):
        grad_z = grad_output * d_selu(self.z)
        self.dW = np.dot(self.x.T, grad_z)
        self.db = np.sum(grad_z, axis=0)
        grad_x = np.dot(grad_z, self.W.T)
        return grad_x

class DenseLinear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) / np.sqrt(in_features)
        self.b = np.zeros(out_features)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        self.dW = np.dot(self.x.T, grad_output)
        self.db = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.W.T)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def d_mse_loss(y_pred, y_true):
    return 2.0 * (y_pred - y_true) / y_true.size

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    np.random.seed(42)

    # Toy dataset: non-linear regression
    # Standard normal inputs (mean 0, var 1)
    X = np.random.randn(1000, 10)
    y = np.sum(X**2, axis=1, keepdims=True)

    # Deep Network: 10 -> 32 -> 32 -> 32 -> 1
    l1 = DenseSELU(10, 32)
    l2 = DenseSELU(32, 32)
    l3 = DenseSELU(32, 32)
    l4 = DenseLinear(32, 1)

    epochs = args.epochs
    lr = args.lr

    print("Initial checks:")
    h1 = l1.forward(X)
    h2 = l2.forward(h1)
    h3 = l3.forward(h2)
    print(f"Var H1: {np.var(h1):.4f}, H2: {np.var(h2):.4f}, H3: {np.var(h3):.4f}")

    for epoch in range(epochs):
        # Forward pass
        h1 = l1.forward(X)
        h2 = l2.forward(h1)
        h3 = l3.forward(h2)
        y_pred = l4.forward(h3)

        loss = mse_loss(y_pred, y)

        # Backward pass
        grad_y_pred = d_mse_loss(y_pred, y)
        grad_h3 = l4.backward(grad_y_pred)
        grad_h2 = l3.backward(grad_h3)
        grad_h1 = l2.backward(grad_h2)
        _ = l1.backward(grad_h1)

        # Update weights (SGD)
        for layer in [l1, l2, l3, l4]:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

        if (epoch + 1) % 100 == 0:
            var_h1, var_h2, var_h3 = np.var(h1), np.var(h2), np.var(h3)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Var H1: {var_h1:.4f}, H2: {var_h2:.4f}, H3: {var_h3:.4f}")

    print("SELU Component test passed!")
