import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        self.dW[:] = self.X.T @ dZ
        self.db[:] = np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dZ):
        return dZ * (self.X > 0)

def softmax(x, axis=-1):
    exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exps / np.sum(exps, axis=axis, keepdims=True)

class MediumScaleModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layers = [
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db

def train():
    np.random.seed(42)
    N = 2000
    X = np.random.randint(0, 2, size=(N, 10))
    y = np.sum(X, axis=1) % 2

    X_scaled = (X - 0.5) * 2.0

    model = MediumScaleModel(10, 128, 2)

    epochs = 400
    lr = 0.5

    print("Training Medium-Scale Model on 10-bit parity task...")
    for epoch in range(epochs):
        logits = model.forward(X_scaled)

        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-9))

        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1
        dlogits /= N

        model.backward(dlogits)
        model.update(lr)

        if epoch % 50 == 0:
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y)
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    preds = np.argmax(model.forward(X_scaled), axis=1)
    acc = np.mean(preds == y)
    print(f"Final Accuracy: {acc:.4f}")
    if acc > 0.9:
        print("Medium-scale model successfully trained and solved the parity task.")
    else:
        print("Model failed to fully solve the parity task.")

if __name__ == '__main__':
    train()
