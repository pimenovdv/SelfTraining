import numpy as np

class BetaVAE:
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=4.0, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.lr = lr

        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)

        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_mu = np.zeros(latent_dim)

        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_logvar = np.zeros(latent_dim)

        self.W2 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)

        self.W3 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b3 = np.zeros(input_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # Encoder
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        self.mu = np.dot(self.a1, self.W_mu) + self.b_mu
        self.logvar = np.dot(self.a1, self.W_logvar) + self.b_logvar

        # Reparameterization
        self.std = np.exp(0.5 * self.logvar)
        self.eps = np.random.randn(*self.std.shape)
        self.z = self.mu + self.eps * self.std

        # Decoder
        self.z2 = np.dot(self.z, self.W2) + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.out = self.sigmoid(self.z3)

        return self.out, self.mu, self.logvar

    def compute_loss(self, X, out, mu, logvar):
        # BCE with clipping to prevent log(0)
        out = np.clip(out, 1e-15, 1 - 1e-15)
        BCE = -np.sum(X * np.log(out) + (1 - X) * np.log(1 - out)) / X.shape[0]

        KLD = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / X.shape[0]

        return BCE + self.beta * KLD, BCE, KLD

    def backward(self, X):
        m = X.shape[0]

        # Derivative of BCE w.r.t out
        out = np.clip(self.out, 1e-15, 1 - 1e-15)
        d_out = -(X / out) + ((1 - X) / (1 - out))

        # Derivative of Sigmoid
        d_z3 = d_out * out * (1 - out)

        d_W3 = np.dot(self.a2.T, d_z3) / m
        d_b3 = np.sum(d_z3, axis=0) / m

        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2 * self.relu_deriv(self.z2)

        d_W2 = np.dot(self.z.T, d_z2) / m
        d_b2 = np.sum(d_z2, axis=0) / m

        d_z = np.dot(d_z2, self.W2.T)

        # KLD gradients
        d_mu = (d_z + self.beta * self.mu) / m
        d_logvar = (d_z * 0.5 * self.eps * self.std + self.beta * 0.5 * (np.exp(self.logvar) - 1)) / m

        d_W_mu = np.dot(self.a1.T, d_mu)
        d_b_mu = np.sum(d_mu, axis=0)

        d_W_logvar = np.dot(self.a1.T, d_logvar)
        d_b_logvar = np.sum(d_logvar, axis=0)

        d_a1 = np.dot(d_mu, self.W_mu.T) + np.dot(d_logvar, self.W_logvar.T)
        d_z1 = d_a1 * self.relu_deriv(self.z1)

        d_W1 = np.dot(X.T, d_z1) / m
        d_b1 = np.sum(d_z1, axis=0) / m

        # Update weights (SGD)
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

        self.W_mu -= self.lr * d_W_mu
        self.b_mu -= self.lr * d_b_mu
        self.W_logvar -= self.lr * d_W_logvar
        self.b_logvar -= self.lr * d_b_logvar

        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

def main():
    print("Initializing Beta-VAE...")
    np.random.seed(42)
    input_dim = 10
    hidden_dim = 20
    latent_dim = 5
    beta = 4.0

    model = BetaVAE(input_dim, hidden_dim, latent_dim, beta, lr=0.01)

    # Generate some dummy synthetic data
    # Continuous data in [0, 1]
    X = np.random.rand(100, input_dim)

    epochs = 50
    for epoch in range(epochs):
        out, mu, logvar = model.forward(X)
        loss, bce, kld = model.compute_loss(X, out, mu, logvar)
        model.backward(X)

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, BCE: {bce:.4f}, KLD: {kld:.4f}")

    print("Beta-VAE training completed successfully.")

if __name__ == '__main__':
    main()
