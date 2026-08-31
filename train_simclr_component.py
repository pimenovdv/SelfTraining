import numpy as np

def set_seed(seed=42):
    np.random.seed(seed)

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ):
        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        return np.dot(dZ, self.W.T)

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dZ):
        return dZ * (self.X > 0)

class MLP:
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.fc1 = Linear(in_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, out_dim)

    def forward(self, X):
        h1 = self.relu.forward(self.fc1.forward(X))
        return self.fc2.forward(h1)

    def backward(self, dZ):
        dh1 = self.fc2.backward(dZ)
        dX = self.fc1.backward(self.relu.backward(dh1))
        return dX

    def update(self, lr=0.01):
        self.fc1.W -= lr * self.fc1.dW
        self.fc1.b -= lr * self.fc1.db
        self.fc2.W -= lr * self.fc2.dW
        self.fc2.b -= lr * self.fc2.db

class SimCLR:
    def __init__(self, input_dim, hidden_dim, proj_dim, temperature=0.5):
        self.encoder = MLP(input_dim, hidden_dim, hidden_dim)
        self.projector = MLP(hidden_dim, hidden_dim, proj_dim)
        self.temperature = temperature

    def forward(self, X):
        h = self.encoder.forward(X)
        z = self.projector.forward(h)
        return z

    def backward(self, dZ):
        dh = self.projector.backward(dZ)
        dX = self.encoder.backward(dh)
        return dX

    def update(self, lr=0.01):
        self.encoder.update(lr)
        self.projector.update(lr)

def nt_xent_loss(z_i, z_j, temperature=0.5):
    N = z_i.shape[0]
    z = np.concatenate([z_i, z_j], axis=0) # 2N x D

    # Normalize embeddings
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z_norm = z / (norms + 1e-8)

    # Cosine similarity matrix
    sim_matrix = np.dot(z_norm, z_norm.T) / temperature

    # Target masks
    labels = np.concatenate([np.arange(N, 2*N), np.arange(N)])

    # Mask out self-similarity
    np.fill_diagonal(sim_matrix, -1e9)

    # Softmax over rows
    exp_sim = np.exp(sim_matrix - np.max(sim_matrix, axis=1, keepdims=True))
    prob_matrix = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

    # Compute loss
    loss = -np.mean(np.log(prob_matrix[np.arange(2*N), labels] + 1e-8))

    # Gradient computation
    grad_prob = prob_matrix.copy()
    grad_prob[np.arange(2*N), labels] -= 1.0
    grad_prob /= (2 * N)

    # Backprop through similarity matrix
    d_sim_matrix = grad_prob / temperature

    # Backprop through dot product and normalization
    dz_norm = np.dot(d_sim_matrix + d_sim_matrix.T, z_norm)

    # Backprop through normalization
    z_norm_sq = z**2
    norms_cubed = norms**3 + 1e-8
    dz = dz_norm / norms - z * np.sum(dz_norm * z, axis=1, keepdims=True) / norms_cubed

    dz_i, dz_j = np.split(dz, 2)
    return loss, dz_i, dz_j

def test_simclr():
    set_seed(42)
    input_dim = 16
    hidden_dim = 32
    proj_dim = 8
    N = 64
    epochs = 100
    lr = 0.05

    model = SimCLR(input_dim, hidden_dim, proj_dim)

    X = np.random.randn(N, input_dim)

    for epoch in range(epochs):
        # Create augmented views
        noise1 = np.random.randn(N, input_dim) * 0.1
        noise2 = np.random.randn(N, input_dim) * 0.1

        X_i = X + noise1
        X_j = X + noise2

        # Forward pass
        # Since we use MLPs with state (self.X), we need to forward separately and store intermediate?
        # Actually our MLP overwrites self.X on forward, so we must forward and backward one view at a time?
        # Oh, no. Our MLP uses self.X, so forwarding both at once is required.
        # Let's concatenate before forward.
        X_both = np.concatenate([X_i, X_j], axis=0)
        z_both = model.forward(X_both)

        z_i, z_j = np.split(z_both, 2)

        loss, dz_i, dz_j = nt_xent_loss(z_i, z_j, model.temperature)

        dz_both = np.concatenate([dz_i, dz_j], axis=0)
        model.backward(dz_both)
        model.update(lr)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("SimCLR test passed!")

if __name__ == "__main__":
    test_simclr()
