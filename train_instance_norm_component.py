import numpy as np

class InstanceNorm:
    def __init__(self, channels):
        self.gamma = np.ones((1, channels, 1, 1))
        self.beta = np.zeros((1, channels, 1, 1))
        self.eps = 1e-5
        self.mean = None
        self.var = None
        self.x_centered = None
        self.std_inv = None
        self.x_norm = None

    def forward(self, x):
        self.mean = np.mean(x, axis=(2, 3), keepdims=True)
        self.var = np.var(x, axis=(2, 3), keepdims=True)
        self.x_centered = x - self.mean
        self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
        self.x_norm = self.x_centered * self.std_inv
        out = self.gamma * self.x_norm + self.beta
        return out

    def backward(self, dout, x):
        N, C, H, W = x.shape
        D = H * W

        dgamma = np.sum(dout * self.x_norm, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * (self.std_inv ** 3), axis=(2, 3), keepdims=True)
        dmean = np.sum(dx_norm * -self.std_inv, axis=(2, 3), keepdims=True) + dvar * np.mean(-2.0 * self.x_centered, axis=(2, 3), keepdims=True)

        dx = dx_norm * self.std_inv + dvar * 2.0 * self.x_centered / D + dmean / D

        return dx, dgamma, dbeta

def test_component():
    np.random.seed(42)
    N, C, H, W = 4, 3, 8, 8
    x = np.random.randn(N, C, H, W)
    y_true = np.random.randn(N, C, H, W)

    model = InstanceNorm(C)

    learning_rate = 0.1
    epochs = 1000

    for epoch in range(epochs):
        # Forward pass
        y_pred = model.forward(x)

        # Compute loss (Mean Squared Error)
        loss = np.mean((y_pred - y_true) ** 2)

        # Backward pass
        dout = 2.0 * (y_pred - y_true) / (N * C * H * W)
        dx, dgamma, dbeta = model.backward(dout, x)

        # Update parameters
        model.gamma -= learning_rate * dgamma
        model.beta -= learning_rate * dbeta

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Success")

if __name__ == "__main__":
    test_component()
