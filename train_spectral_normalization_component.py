import numpy as np
import os

class SpectralNormLinear:
    def __init__(self, in_features, out_features, n_power_iterations=1, eps=1e-12):
        self.in_features = in_features
        self.out_features = out_features
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        self.W = np.random.randn(out_features, in_features) / np.sqrt(in_features)
        self.b = np.zeros(out_features)

        self.u = np.random.randn(out_features)
        self.u /= np.linalg.norm(self.u) + eps

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.x = None
        self.W_sn = None

    def power_iteration(self):
        u = self.u
        for _ in range(self.n_power_iterations):
            v = self.W.T @ u
            v /= np.linalg.norm(v) + self.eps
            u = self.W @ v
            u /= np.linalg.norm(u) + self.eps

        self.u = u
        v = self.W.T @ u
        sigma = np.dot(u, self.W @ v)
        return sigma, u, v

    def forward(self, x, training=True):
        self.x = x

        if training:
            sigma, u, v = self.power_iteration()
            self.W_sn = self.W / sigma
            self.sigma = sigma
            self.u_cache = u
            self.v_cache = v
        else:
            if self.W_sn is None:
                 sigma, _, _ = self.power_iteration()
                 self.W_sn = self.W / sigma

        out = x @ self.W_sn.T + self.b
        return out

    def backward(self, dout):
        self.db = np.sum(dout, axis=0)
        dW_sn = dout.T @ self.x

        lambda_val = np.sum(dW_sn * self.W)
        dsigma_dW = np.outer(self.u_cache, self.v_cache)

        self.dW = (dW_sn - lambda_val * dsigma_dW / self.sigma) / self.sigma

        dx = dout @ self.W_sn
        return dx

def train_spectral_normalization():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    Y_true = (X.sum(axis=1) > 0).astype(float).reshape(-1, 1)

    layer1 = SpectralNormLinear(10, 5)
    layer2 = SpectralNormLinear(5, 1)

    learning_rate = 0.5
    losses = []

    for epoch in range(200):
        h1 = layer1.forward(X)
        h1_relu = np.maximum(0, h1)
        logits = layer2.forward(h1_relu)

        probs = 1 / (1 + np.exp(-logits))

        loss = -np.mean(Y_true * np.log(probs + 1e-12) + (1 - Y_true) * np.log(1 - probs + 1e-12))
        losses.append(loss)

        dprobs = (probs - Y_true) / (probs * (1 - probs) + 1e-12) / len(X)
        dlogits = probs * (1 - probs) * dprobs

        dh1_relu = layer2.backward(dlogits)
        dh1 = dh1_relu * (h1 > 0)
        dx = layer1.backward(dh1)

        layer1.W -= learning_rate * layer1.dW
        layer1.b -= learning_rate * layer1.db
        layer2.W -= learning_rate * layer2.dW
        layer2.b -= learning_rate * layer2.db

    return losses[0] > losses[-1]

if __name__ == "__main__":
    success = train_spectral_normalization()
    print("Spectral Normalization component training successful?" if success else "Failed")

    doc_content = r"""# Experiment: Spectral Normalization Component

**Script:** `train_spectral_normalization_component.py`

## Objective
To implement and mathematically formalize Spectral Normalization for deep neural networks using power iteration, enabling Lipschitz continuity for more stable training.

## Methodology
1.  **Component:** SpectralNormLinear
2.  **Algorithm:** Power Iteration to find the largest singular value $\sigma$ of weight matrix $W$.
3.  **Forward Pass:** Weight matrix is scaled by $1/\sigma$.
4.  **Backward Pass:** Exact gradient computation incorporating the derivative of $\sigma$ with respect to $W$.
5.  **Task:** Binary classification using a two-layer spectral normalized network.

## Results
- **Success:** """ + ('Yes' if success else 'No') + r"""

## Conclusion
Spectral Normalization provides an effective way to enforce Lipschitz constraints, mathematically grounding regularization techniques for advanced generative and discriminative models.
"""

    os.makedirs("docs", exist_ok=True)
    existing_docs = [f for f in os.listdir("docs") if f.endswith(".md")]
    next_index = len(existing_docs) + 1
    doc_filename = f"docs/{next_index:04d}_train_spectral_normalization_component.md"

    with open(doc_filename, "w") as f:
        f.write(doc_content)
    print(f"Documented at {doc_filename}")
