import numpy as np
import os

class EBM:
    def __init__(self, input_dim, hidden_dim):
        # Initialize weights
        self.w1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, 1) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros(1)

    def energy(self, x):
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        return np.dot(h, self.w2) + self.b2

    def energy_grad_x(self, x):
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        dh_dz = 1.0 - h**2
        # dz: (batch_size, hidden_dim)
        dz = np.dot(np.ones((x.shape[0], 1)), self.w2.T) * dh_dz
        return np.dot(dz, self.w1.T)

    def grad_params(self, x):
        batch_size = x.shape[0]
        h = np.tanh(np.dot(x, self.w1) + self.b1)

        dw2_grad = np.dot(h.T, np.ones((batch_size, 1))) / batch_size
        db2_grad = np.array([1.0])

        dh_dz = 1.0 - h**2
        dz = np.dot(np.ones((batch_size, 1)), self.w2.T) * dh_dz

        dw1_grad = np.dot(x.T, dz) / batch_size
        db1_grad = np.mean(dz, axis=0)

        return dw1_grad, db1_grad, dw2_grad, db2_grad

def langevin_dynamics(ebm, x, steps, alpha):
    for _ in range(steps):
        grad_x = ebm.energy_grad_x(x)
        noise = np.random.randn(*x.shape)
        x = x - (alpha / 2.0) * grad_x + np.sqrt(alpha) * noise
    return x

def train():
    np.random.seed(42)
    input_dim = 2
    hidden_dim = 32
    ebm = EBM(input_dim, hidden_dim)

    # Simple target distribution: a ring
    batch_size = 64
    lr = 0.01

    for epoch in range(100):
        # Real data
        angles = np.random.rand(batch_size) * 2 * np.pi
        r = 2.0 + np.random.randn(batch_size) * 0.1
        x_pos = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)

        # Fake data
        x_neg = np.random.randn(batch_size, input_dim) * 3
        x_neg = langevin_dynamics(ebm, x_neg, steps=20, alpha=0.1)

        # Gradients
        dw1_pos, db1_pos, dw2_pos, db2_pos = ebm.grad_params(x_pos)
        dw1_neg, db1_neg, dw2_neg, db2_neg = ebm.grad_params(x_neg)

        # Update
        ebm.w1 -= lr * (dw1_pos - dw1_neg)
        ebm.b1 -= lr * (db1_pos - db1_neg)
        ebm.w2 -= lr * (dw2_pos - dw2_neg)
        ebm.b2 -= lr * (db2_pos - db2_neg)

        if epoch % 20 == 0:
            pos_energy = np.mean(ebm.energy(x_pos))
            neg_energy = np.mean(ebm.energy(x_neg))
            print(f"Epoch {epoch}: Pos Energy {pos_energy:.4f}, Neg Energy {neg_energy:.4f}")

    print("Success")

if __name__ == "__main__":
    train()
