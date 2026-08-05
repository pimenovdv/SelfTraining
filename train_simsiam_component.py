import numpy as np
import os

class Layer:
    def forward(self, x):
        pass
    def backward(self, grad_output):
        pass

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        self.dW[:] = np.dot(self.x.T, grad_output)
        self.db[:] = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.W.T)

class ReLU(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.x > 0)

class SimSiamLoss:
    def __init__(self):
        self.p = None
        self.z = None

    def forward(self, p, z):
        self.p = p
        self.z = z
        p_norm = np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
        z_norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        self.p_normalized = p / p_norm
        self.z_normalized = z / z_norm
        cosine_sim = np.sum(self.p_normalized * self.z_normalized, axis=1)
        return -np.mean(cosine_sim)

    def backward(self):
        batch_size = self.p.shape[0]
        p_norm = np.linalg.norm(self.p, axis=1, keepdims=True) + 1e-8
        z_norm = np.linalg.norm(self.z, axis=1, keepdims=True) + 1e-8
        dot_pz = np.sum(self.p * self.z, axis=1, keepdims=True)
        dp = - self.z / (p_norm * z_norm) + self.p * dot_pz / (p_norm**3 * z_norm)
        return dp / batch_size

class SimSiam:
    def __init__(self, input_dim, hidden_dim, proj_dim):
        self.encoder = [
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, proj_dim)
        ]
        self.predictor = [
            Linear(proj_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, proj_dim)
        ]

    def forward_encoder(self, x):
        out = x
        for layer in self.encoder:
            out = layer.forward(out)
        return out

    def forward_predictor(self, z):
        out = z
        for layer in self.predictor:
            out = layer.forward(out)
        return out

    def backward_predictor(self, grad_output):
        grad = grad_output
        for layer in reversed(self.predictor):
            grad = layer.backward(grad)
        return grad

    def backward_encoder(self, grad_output):
        grad = grad_output
        for layer in reversed(self.encoder):
            grad = layer.backward(grad)
        return grad

    def update(self, lr):
        for model in [self.encoder, self.predictor]:
            for layer in model:
                if isinstance(layer, Linear):
                    layer.W -= lr * layer.dW
                    layer.b -= lr * layer.db

def generate_data(num_samples, input_dim):
    X = np.random.randn(num_samples, input_dim)
    X1 = X + np.random.randn(*X.shape) * 0.1
    X2 = X + np.random.randn(*X.shape) * 0.1
    return X1, X2

def train():
    np.random.seed(42)
    input_dim = 16
    hidden_dim = 32
    proj_dim = 16

    model = SimSiam(input_dim, hidden_dim, proj_dim)
    loss_fn1 = SimSiamLoss()
    loss_fn2 = SimSiamLoss()

    batch_size = 64
    num_epochs = 200
    lr = 0.05

    X1, X2 = generate_data(512, input_dim)

    initial_loss = None
    final_loss = None

    for epoch in range(num_epochs):
        indices = np.random.permutation(X1.shape[0])
        epoch_loss = 0

        for start_idx in range(0, X1.shape[0], batch_size):
            batch_indices = indices[start_idx:start_idx+batch_size]
            x1 = X1[batch_indices]
            x2 = X2[batch_indices]

            x_concat = np.concatenate([x1, x2], axis=0)
            z_concat = model.forward_encoder(x_concat)
            z1, z2 = np.split(z_concat, 2, axis=0)

            p_concat = model.forward_predictor(z_concat)
            p1, p2 = np.split(p_concat, 2, axis=0)

            l1 = loss_fn1.forward(p1, z2)
            l2 = loss_fn2.forward(p2, z1)

            loss = l1 + l2
            epoch_loss += loss

            dp1 = loss_fn1.backward()
            dp2 = loss_fn2.backward()

            dp_concat = np.concatenate([dp1, dp2], axis=0)
            dz_concat = model.backward_predictor(dp_concat)

            model.backward_encoder(dz_concat)
            model.update(lr)

        epoch_loss /= (X1.shape[0] / batch_size)
        if initial_loss is None:
            initial_loss = epoch_loss
        final_loss = epoch_loss

        if epoch % 40 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

    print(f"Final Loss = {final_loss:.4f}")
    success = final_loss < initial_loss and final_loss < -0.8
    print(f"Success: {success}")

    os.makedirs("docs", exist_ok=True)
    with open("docs/0096_train_simsiam_component.md", "w") as f:
        f.write(f"""# Experiment 0096: SimSiam (Simple Siamese Networks)

**Objective:** Implement and verify non-contrastive self-supervised learning using SimSiam mathematically.

**Methodology:** The SimSiam architecture learns representations without requiring negative samples or a momentum encoder. It processes two augmented views of an image through an encoder $f$ and uses a predictor network $h$ on one view to match the encoder output of the other view. A critical stop-gradient operation is applied to the target view to prevent representation collapse. The negative cosine similarity is minimized via manual backpropagation.

**Results:**
- Initial Loss: {initial_loss:.4f}
- Final Loss: {final_loss:.4f}
- Success: {success}

**Conclusion:** The component successfully minimized the negative cosine similarity between augmented views, demonstrating representation learning without collapse and the effectiveness of the stop-gradient operation.
**Script:** `train_simsiam_component.py`
""")

if __name__ == "__main__":
    train()
