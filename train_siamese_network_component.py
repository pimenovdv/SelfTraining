import numpy as np
import os

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class SiameseNetwork:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, latent_dim))

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = z2
        return z1, a1, z2, a2

    def get_gradients(self, x, z1, a1, z2, a2, dL_da2):
        dL_dz2 = dL_da2
        dW2 = np.dot(a1.T, dL_dz2)
        db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * relu_deriv(z1)

        dW1 = np.dot(x.T, dL_dz1)
        db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_weights(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

def contrastive_loss(a2_1, a2_2, label, margin=1.0):
    diff = a2_1 - a2_2
    dist_sq = np.sum(diff**2, axis=1, keepdims=True)
    dist = np.sqrt(dist_sq + 1e-9)

    loss_similar = label * dist_sq
    loss_dissimilar = (1 - label) * np.maximum(0, margin - dist)**2

    loss = np.mean(loss_similar + loss_dissimilar)

    grad_similar = 2 * diff * label
    grad_dissimilar_dist = -2 * np.maximum(0, margin - dist) * (1 - label)
    grad_dissimilar = grad_dissimilar_dist * (diff / dist)

    dL_da2_1 = (grad_similar + grad_dissimilar) / a2_1.shape[0]
    dL_da2_2 = -dL_da2_1

    return loss, dL_da2_1, dL_da2_2

def generate_data(num_samples=1000):
    np.random.seed(42)
    X1_class0 = np.random.randn(num_samples // 4, 10) + 2
    X1_class1 = np.random.randn(num_samples // 4, 10) - 2
    X2_class0 = np.random.randn(num_samples // 4, 10) + 2
    X2_class1 = np.random.randn(num_samples // 4, 10) - 2

    X1_sim = np.vstack([X1_class0, X1_class1])
    X2_sim = np.vstack([X2_class0, X2_class1])
    y_sim = np.ones((num_samples // 2, 1))

    X1_diff = np.vstack([X1_class0, X1_class1])
    X2_diff = np.vstack([X2_class1, X2_class0])
    y_diff = np.zeros((num_samples // 2, 1))

    X1 = np.vstack([X1_sim, X1_diff])
    X2 = np.vstack([X2_sim, X2_diff])
    y = np.vstack([y_sim, y_diff])

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    return X1[indices], X2[indices], y[indices]

def train():
    print("Training Siamese Network for Metric Learning...")
    X1, X2, y = generate_data(1000)

    model = SiameseNetwork(10, 16, 4)
    lr = 0.05
    epochs = 100
    batch_size = 32

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X1), batch_size):
            x1_batch = X1[i:i+batch_size]
            x2_batch = X2[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            z1_1, a1_1, z2_1, a2_1 = model.forward(x1_batch)
            z1_2, a1_2, z2_2, a2_2 = model.forward(x2_batch)

            loss, dL_da2_1, dL_da2_2 = contrastive_loss(a2_1, a2_2, y_batch, margin=1.0)
            epoch_loss += loss * len(x1_batch)

            grads_1 = model.get_gradients(x1_batch, z1_1, a1_1, z2_1, a2_1, dL_da2_1)
            grads_2 = model.get_gradients(x2_batch, z1_2, a1_2, z2_2, a2_2, dL_da2_2)

            total_grads = [g1 + g2 for g1, g2 in zip(grads_1, grads_2)]
            model.update_weights(total_grads, lr)

        epoch_loss /= len(X1)
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train()
