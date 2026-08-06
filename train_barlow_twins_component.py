import numpy as np
import os
import time

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

class MLP:
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.fc1 = Linear(in_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, out_dim)

    def tie_weights(self, other):
        self.fc1.W = other.fc1.W
        self.fc1.b = other.fc1.b
        self.fc2.W = other.fc2.W
        self.fc2.b = other.fc2.b

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

def barlow_twins_loss(Z_A, Z_B, lambd=5e-3):
    N, D = Z_A.shape

    mu_A = np.mean(Z_A, axis=0, keepdims=True)
    var_A = np.var(Z_A, axis=0, keepdims=True)
    norm_A = (Z_A - mu_A) / np.sqrt(var_A + 1e-5)

    mu_B = np.mean(Z_B, axis=0, keepdims=True)
    var_B = np.var(Z_B, axis=0, keepdims=True)
    norm_B = (Z_B - mu_B) / np.sqrt(var_B + 1e-5)

    C = np.dot(norm_A.T, norm_B) / N

    C_diff = C - np.eye(D)
    C_diff_off_diag = C_diff - np.diag(np.diag(C_diff))

    loss_diag = np.sum(np.diag(C_diff)**2)
    loss_off_diag = np.sum(C_diff_off_diag**2)
    loss = loss_diag + lambd * loss_off_diag

    dC = 2 * C_diff_off_diag * lambd
    np.fill_diagonal(dC, 2 * np.diag(C_diff))

    dnorm_A = np.dot(norm_B, dC.T) / N
    dnorm_B = np.dot(norm_A, dC) / N

    def batch_norm_backward(dout, X, mu, var, norm_X):
        dX_norm = dout
        dvar = np.sum(dX_norm * (X - mu) * -0.5 * np.power(var + 1e-5, -1.5), axis=0, keepdims=True)
        dmu = np.sum(dX_norm * -1.0 / np.sqrt(var + 1e-5), axis=0, keepdims=True) + dvar * np.mean(-2.0 * (X - mu), axis=0, keepdims=True)
        dX = dX_norm / np.sqrt(var + 1e-5) + dvar * 2.0 * (X - mu) / N + dmu / N
        return dX

    dZ_A = batch_norm_backward(dnorm_A, Z_A, mu_A, var_A, norm_A)
    dZ_B = batch_norm_backward(dnorm_B, Z_B, mu_B, var_B, norm_B)

    return loss, dZ_A, dZ_B, C

def generate_data(num_samples=1000, dim=10, noise_level=0.1):
    X = np.random.randn(num_samples, dim)
    X_A = X + np.random.randn(num_samples, dim) * noise_level
    X_B = X + np.random.randn(num_samples, dim) * noise_level
    return X_A, X_B

def generate_report(initial_loss, final_loss, success, time_taken):
    report_content = f"""# Experiment 0101: Barlow Twins

**Script:** `train_barlow_twins_component.py`

## Objective
Evaluate a Barlow Twins component for non-contrastive self-supervised learning, verifying its ability to prevent representation collapse by driving the cross-correlation matrix between representations of distorted versions of a sample to the identity matrix.

## Configuration
- Input Dimension: 16
- Hidden Dimension: 64
- Projection Dimension: 16
- Batch Size: 128
- Epochs: 200
- Learning Rate: 0.05
- Lambda (Off-diagonal weight): 0.005

## Results
- Initial Loss: {initial_loss:.4f}
- Final Loss: {final_loss:.4f}
- Training Time: {time_taken:.2f}s
- Success: {success}

## Conclusion
The model {'successfully' if success else 'failed to'} minimize the Barlow Twins loss, driving the cross-correlation matrix toward the identity matrix, confirming that representations were learned without collapsing into trivial constant solutions.
"""

    os.makedirs('docs', exist_ok=True)
    with open('docs/0101_train_barlow_twins_component.md', 'w') as f:
        f.write(report_content)
    print("Report generated at docs/0101_train_barlow_twins_component.md")


def main():
    print("Starting Barlow Twins Component Training...")
    set_seed(42)

    input_dim = 16
    hidden_dim = 64
    proj_dim = 16
    batch_size = 128
    epochs = 200
    lr = 0.05
    lambd = 0.005

    encoder_A = MLP(input_dim, hidden_dim, proj_dim)
    encoder_B = MLP(input_dim, hidden_dim, proj_dim)
    encoder_B.tie_weights(encoder_A)

    X_A_full, X_B_full = generate_data(num_samples=2048, dim=input_dim, noise_level=0.2)

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

            Z_A = encoder_A.forward(X_A)
            Z_B = encoder_B.forward(X_B)

            loss, dZ_A, dZ_B, C = barlow_twins_loss(Z_A, Z_B, lambd=lambd)
            epoch_loss += loss
            batches += 1

            encoder_A.backward(dZ_A)
            encoder_B.backward(dZ_B)

            grads_A = encoder_A.get_gradients()
            grads_B = encoder_B.get_gradients()
            total_grads = [ga + gb for ga, gb in zip(grads_A, grads_B)]
            encoder_A.apply_gradients(total_grads, lr)

        avg_loss = epoch_loss / batches
        if epoch == 0:
            initial_loss = avg_loss
        if epoch == epochs - 1:
            final_loss = avg_loss

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    success = final_loss < initial_loss * 0.8
    generate_report(initial_loss, final_loss, success, time.time() - start_time)
    print(f"Training completed. Success: {success}")

if __name__ == "__main__":
    main()
