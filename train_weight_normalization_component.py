import numpy as np
import os

class WeightNormLinear:
    def __init__(self, in_features, out_features, eps=1e-8):
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.v = np.random.randn(out_features, in_features) / np.sqrt(in_features)
        self.g = np.linalg.norm(self.v, axis=1, keepdims=True)
        self.b = np.zeros(out_features)

        self.dv = np.zeros_like(self.v)
        self.dg = np.zeros_like(self.g)
        self.db = np.zeros_like(self.b)

        self.x = None
        self.W = None

    def forward(self, x):
        self.x = x
        v_norm = np.linalg.norm(self.v, axis=1, keepdims=True) + self.eps
        self.W = (self.g / v_norm) * self.v
        out = x @ self.W.T + self.b
        return out

    def backward(self, dout):
        self.db = np.sum(dout, axis=0)
        dW = dout.T @ self.x

        v_norm = np.linalg.norm(self.v, axis=1, keepdims=True) + self.eps
        self.dg = np.sum(dW * (self.v / v_norm), axis=1, keepdims=True)

        dv_term1 = self.g / v_norm
        dv_term2 = - (self.g / (v_norm ** 3)) * self.v * np.sum(dW * self.v, axis=1, keepdims=True)

        self.dv = dW * dv_term1 + dv_term2

        dx = dout @ self.W
        return dx

def train_weight_normalization():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    Y_true = (X.sum(axis=1) > 0).astype(float).reshape(-1, 1)

    layer1 = WeightNormLinear(10, 5)
    layer2 = WeightNormLinear(5, 1)

    learning_rate = 0.1
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

        layer1.v -= learning_rate * layer1.dv
        layer1.g -= learning_rate * layer1.dg
        layer1.b -= learning_rate * layer1.db
        layer2.v -= learning_rate * layer2.dv
        layer2.g -= learning_rate * layer2.dg
        layer2.b -= learning_rate * layer2.db

    return losses[0] > losses[-1]

if __name__ == "__main__":
    success = train_weight_normalization()
    print("Weight Normalization component training successful?" if success else "Failed")

    doc_content = r"""# Experiment: Weight Normalization Component

**Script:** `train_weight_normalization_component.py`

## Objective
To implement and mathematically formalize Weight Normalization, a reparameterization of the weight vectors in a neural network that decouples the length of those weight vectors from their direction.

## Methodology
1.  **Component:** WeightNormLinear
2.  **Algorithm:** Weight matrix $w$ is reparameterized as $w = (g / ||v||) * v$, where $v$ is a parameter vector and $g$ is a scalar parameter.
3.  **Forward Pass:** Calculate norm of $v$, scale by $g/||v||$, and perform linear transformation.
4.  **Backward Pass:** Exact gradient computation for both $v$ and $g$ using the chain rule.
5.  **Task:** Binary classification using a two-layer weight-normalized network.

## Results
- **Success:** """ + ('Yes' if success else 'No') + r"""

## Conclusion
Weight Normalization accelerates convergence similar to batch normalization but does not introduce dependencies between examples in a minibatch, making it suitable for recurrent models and noise-sensitive applications like reinforcement learning.
"""

    os.makedirs("docs", exist_ok=True)
    existing_docs = [f for f in os.listdir("docs") if f.endswith(".md")]
    next_index = len(existing_docs) + 1
    doc_filename = f"docs/{next_index:04d}_train_weight_normalization_component.md"

    with open(doc_filename, "w") as f:
        f.write(doc_content)
    print(f"Documented at {doc_filename}")
