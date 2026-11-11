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

class Encoder:
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.fc1 = Linear(in_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, out_dim)

    def copy_from(self, other, momentum=0.99):
        self.fc1.W = momentum * self.fc1.W + (1 - momentum) * other.fc1.W
        self.fc1.b = momentum * self.fc1.b + (1 - momentum) * other.fc1.b
        self.fc2.W = momentum * self.fc2.W + (1 - momentum) * other.fc2.W
        self.fc2.b = momentum * self.fc2.b + (1 - momentum) * other.fc2.b

    def init_from(self, other):
        self.fc1.W = other.fc1.W.copy()
        self.fc1.b = other.fc1.b.copy()
        self.fc2.W = other.fc2.W.copy()
        self.fc2.b = other.fc2.b.copy()

    def forward(self, X):
        out = self.fc1.forward(X)
        out = self.relu.forward(out)
        out = self.fc2.forward(out)
        # Normalize
        out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
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

class MoCo:
    def __init__(self, in_dim, hidden_dim, out_dim, queue_size=256, m=0.99, T=0.07):
        self.encoder_q = Encoder(in_dim, hidden_dim, out_dim)
        self.encoder_k = Encoder(in_dim, hidden_dim, out_dim)
        self.encoder_k.init_from(self.encoder_q)

        self.queue = np.random.randn(queue_size, out_dim)
        self.queue = self.queue / (np.linalg.norm(self.queue, axis=1, keepdims=True) + 1e-8)

        self.m = m
        self.T = T
        self.queue_size = queue_size
        self.queue_ptr = 0

    def update_key_encoder(self):
        self.encoder_k.copy_from(self.encoder_q, momentum=self.m)

    def enqueue_dequeue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size, :] = keys
        else:
            diff = self.queue_size - ptr
            self.queue[ptr:, :] = keys[:diff, :]
            self.queue[:batch_size - diff, :] = keys[diff:, :]

        self.queue_ptr = (ptr + batch_size) % self.queue_size

    def contrastive_loss(self, q, k):
        N = q.shape[0]
        # Positive logits
        l_pos = np.sum(q * k, axis=1, keepdims=True)
        # Negative logits
        l_neg = np.dot(q, self.queue.T)

        logits = np.concatenate([l_pos, l_neg], axis=1) / self.T

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross entropy loss (labels are all 0, meaning positive pair is class 0)
        loss = -np.mean(np.log(probs[:, 0] + 1e-8))

        # Gradient for q
        dlogits = probs.copy()
        dlogits[:, 0] -= 1
        dlogits /= (N * self.T)

        dq = dlogits[:, 0:1] * k + np.dot(dlogits[:, 1:], self.queue)

        # Backprop through L2 norm roughly
        q_unnormalized = np.dot(self.encoder_q.fc2.X, self.encoder_q.fc2.W) + self.encoder_q.fc2.b
        q_norm = np.linalg.norm(q_unnormalized, axis=1, keepdims=True) + 1e-8
        dq_raw = (dq - q * np.sum(dq * q, axis=1, keepdims=True)) / q_norm

        return loss, dq_raw

def generate_data(num_samples=1000, dim=16, noise_level=0.1):
    X = np.random.randn(num_samples, dim)
    X_A = X + np.random.randn(num_samples, dim) * noise_level
    X_B = X + np.random.randn(num_samples, dim) * noise_level
    return X_A, X_B

def main():
    print("Starting MoCo Component Training...")
    set_seed(42)

    input_dim = 16
    hidden_dim = 64
    proj_dim = 16
    batch_size = 64
    epochs = 50
    lr = 0.01

    moco = MoCo(input_dim, hidden_dim, proj_dim, queue_size=256)

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

            q = moco.encoder_q.forward(X_A)
            k = moco.encoder_k.forward(X_B)

            loss, dq = moco.contrastive_loss(q, k)
            epoch_loss += loss
            batches += 1

            moco.encoder_q.backward(dq)

            grads = moco.encoder_q.get_gradients()
            moco.encoder_q.apply_gradients(grads, lr)

            moco.update_key_encoder()
            moco.enqueue_dequeue(k)

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
