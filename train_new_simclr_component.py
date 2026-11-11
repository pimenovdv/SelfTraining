import numpy as np
import time
import os

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
        self.dW[:] = np.dot(self.X.T, dZ)
        self.db[:] = np.sum(dZ, axis=0, keepdims=True)
        return np.dot(dZ, self.W.T)

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)
    def backward(self, dZ):
        return dZ * (self.X > 0)

class SimCLR_Encoder:
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.fc1 = Linear(in_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, out_dim)

    def forward(self, X):
        out = self.fc1.forward(X)
        out = self.relu.forward(out)
        out = self.fc2.forward(out)
        return out

    def backward(self, dout):
        dout = self.fc2.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.fc1.backward(dout)
        return dout

    def get_gradients(self):
        return (self.fc1.dW, self.fc1.db, self.fc2.dW, self.fc2.db)

    def apply_gradients(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.fc1.W -= lr * dW1
        self.fc1.b -= lr * db1
        self.fc2.W -= lr * dW2
        self.fc2.b -= lr * db2

def simclr_loss(z_i, z_j, temperature=0.5):
    N = z_i.shape[0]

    # Concatenate representations
    z = np.concatenate([z_i, z_j], axis=0) # 2N x D

    # Normalize representations
    z_norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    z_unit = z / z_norm

    # Compute similarity matrix
    sim = np.dot(z_unit, z_unit.T) / temperature # 2N x 2N

    # Mask out self-similarity
    np.fill_diagonal(sim, -np.inf)

    # Compute probabilities
    max_sim = np.max(sim, axis=1, keepdims=True)
    exp_sim = np.exp(sim - max_sim)
    sum_exp_sim = np.sum(exp_sim, axis=1, keepdims=True)
    prob = exp_sim / sum_exp_sim

    # Labels for positive pairs: i and i+N are positive pairs
    pos_sim = np.zeros(2*N)
    for i in range(N):
        pos_sim[i] = sim[i, i+N]
        pos_sim[i+N] = sim[i+N, i]

    pos_prob = np.zeros(2*N)
    for i in range(N):
        pos_prob[i] = prob[i, i+N]
        pos_prob[i+N] = prob[i+N, i]

    loss = -np.mean(np.log(pos_prob + 1e-8))

    # Gradient of loss w.r.t logits (sim)
    dsim = prob.copy()
    for i in range(N):
        dsim[i, i+N] -= 1
        dsim[i+N, i] -= 1
    dsim /= (2 * N)

    # Gradient w.r.t z_unit
    dz_unit = np.dot(dsim + dsim.T, z_unit) / temperature

    # Gradient w.r.t z (backprop through normalization)
    dz = (dz_unit - z_unit * np.sum(dz_unit * z_unit, axis=1, keepdims=True)) / z_norm

    dz_i = dz[:N]
    dz_j = dz[N:]

    return loss, dz_i, dz_j

def generate_data(num_samples=1000, dim=16, noise_level=0.1):
    X = np.random.randn(num_samples, dim)
    X_A = X + np.random.randn(num_samples, dim) * noise_level
    X_B = X + np.random.randn(num_samples, dim) * noise_level
    return X_A, X_B

def main():
    print("Starting SimCLR Component Training...")
    set_seed(42)

    input_dim = 16
    hidden_dim = 64
    proj_dim = 16
    batch_size = 64
    epochs = 50
    lr = 0.05

    encoder = SimCLR_Encoder(input_dim, hidden_dim, proj_dim)

    X_A_full, X_B_full = generate_data(num_samples=1024, dim=input_dim, noise_level=0.2)

    initial_loss = None
    final_loss = None

    start_time = time.time()
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_A_full))
        X_A_full = X_A_full[perm]
        X_B_full = X_B_full[perm]

        epoch_loss = 0
        batches = 0
        for i in range(0, len(X_A_full), batch_size):
            X_A = X_A_full[i:i+batch_size]
            X_B = X_B_full[i:i+batch_size]
            if len(X_A) < batch_size:
                continue

            z_i = encoder.forward(X_A)

            # Save activations for backward pass
            act_X_i = encoder.fc1.X
            act_h_i = encoder.fc2.X

            z_j = encoder.forward(X_B)

            act_X_j = encoder.fc1.X
            act_h_j = encoder.fc2.X

            loss, dz_i, dz_j = simclr_loss(z_i, z_j, temperature=0.5)
            epoch_loss += loss
            batches += 1

            # Backward pass for j
            encoder.fc1.X = act_X_j
            encoder.fc2.X = act_h_j
            encoder.relu.X = act_h_j # approximated as they share the same relu

            # This is slightly tricky, we need to accumulate gradients
            # We'll just do it explicitly
            encoder.fc1.dW.fill(0)
            encoder.fc1.db.fill(0)
            encoder.fc2.dW.fill(0)
            encoder.fc2.db.fill(0)

            encoder.backward(dz_j)
            grads_j = encoder.get_gradients()

            # Backward pass for i
            encoder.fc1.X = act_X_i
            encoder.fc2.X = act_h_i
            encoder.relu.X = act_h_i

            encoder.fc1.dW.fill(0)
            encoder.fc1.db.fill(0)
            encoder.fc2.dW.fill(0)
            encoder.fc2.db.fill(0)

            encoder.backward(dz_i)
            grads_i = encoder.get_gradients()

            total_grads = tuple(gj + gi for gj, gi in zip(grads_j, grads_i))
            encoder.apply_gradients(total_grads, lr)

        avg_loss = epoch_loss / batches
        if epoch == 0:
            initial_loss = avg_loss
        if epoch == epochs - 1:
            final_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    success = final_loss < initial_loss * 0.95
    print(f"Training completed. Final loss: {final_loss:.4f}, Success: {success}")

if __name__ == "__main__":
    main()
