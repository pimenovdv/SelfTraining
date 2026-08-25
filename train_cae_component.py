import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class ContractiveAutoencoder:
    def __init__(self, input_dim, hidden_dim, lam=1e-4, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lam = lam
        self.lr = lr

        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, input_dim))

    def train(self, X, epochs=1000):
        loss = 0.0
        for epoch in range(epochs):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            h = sigmoid(z1)

            z2 = np.dot(h, self.W2) + self.b2
            x_hat = sigmoid(z2)

            # Reconstruction loss (MSE)
            recon_loss = np.mean((x_hat - X) ** 2)

            # Contractive penalty
            dh = h * (1 - h)
            W_sum_sq = np.sum(self.W1 ** 2, axis=0)
            contractive_penalty = np.mean(np.sum((dh ** 2) * W_sum_sq, axis=1))

            loss = recon_loss + self.lam * contractive_penalty

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f} (Recon: {recon_loss:.4f}, Penalty: {contractive_penalty:.4f})")

            # Backward pass
            batch_size = X.shape[0]

            dx_hat = 2 * (x_hat - X) / batch_size
            dz2 = dx_hat * x_hat * (1 - x_hat)
            dW2 = np.dot(h.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dh_recon = np.dot(dz2, self.W2.T)

            dP_dh = 2 * dh * (1 - 2 * h) * W_sum_sq
            dh_penalty = self.lam * dP_dh / batch_size

            dz1 = (dh_recon + dh_penalty) * dh

            dW1 = np.dot(X.T, dz1)
            dW1_penalty = self.lam * 2 * self.W1 * np.mean(dh ** 2, axis=0)
            dW1 += dW1_penalty

            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Update weights
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

        return loss

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 10)
    cae = ContractiveAutoencoder(input_dim=10, hidden_dim=5, lam=1e-3, lr=0.1)
    final_loss = cae.train(X, epochs=1000)
    print("Training finished. Final Loss:", final_loss)
