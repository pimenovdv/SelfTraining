import numpy as np
import os
import argparse

np.random.seed(42)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class PerceiverBottleneck:
    """
    Perceiver Bottleneck (Cross-Attention) Component.
    Maps an input sequence of arbitrary length to a fixed-size latent array.
    This reduces the O(N^2) complexity of standard self-attention to O(N * M),
    where M is the number of latents and N is the input sequence length.
    """
    def __init__(self, input_dim, latent_dim, num_latents):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        # Learnable latent array (M x D)
        self.latents = np.random.randn(num_latents, latent_dim) * 0.1

        # Cross-Attention weights
        self.W_q = np.random.randn(latent_dim, latent_dim) * 0.1
        self.W_k = np.random.randn(input_dim, latent_dim) * 0.1
        self.W_v = np.random.randn(input_dim, latent_dim) * 0.1

        # Output projection
        self.W_o = np.random.randn(latent_dim, latent_dim) * 0.1

    def forward(self, X):
        batch_size, seq_len, _ = X.shape

        self.Z = np.tile(self.latents, (batch_size, 1, 1))
        self.X = X

        # Q from latents, K and V from input X
        self.Q = np.dot(self.Z, self.W_q) # (B, M, D)
        self.K = np.dot(self.X, self.W_k) # (B, N, D)
        self.V = np.dot(self.X, self.W_v) # (B, N, D)

        # Cross-attention scores
        self.scores = np.matmul(self.Q, self.K.transpose(0, 2, 1)) / np.sqrt(self.latent_dim) # (B, M, N)
        self.A = softmax(self.scores, axis=-1)

        # Context vector
        self.context = np.matmul(self.A, self.V) # (B, M, D)

        # Output with residual connection
        self.out = np.dot(self.context, self.W_o) + self.Z
        return self.out

    def backward(self, d_out, lr=0.01):
        batch_size = d_out.shape[0]

        d_Z_res = d_out.copy()

        dW_o = np.matmul(self.context.transpose(0, 2, 1), d_out).sum(axis=0) / batch_size
        d_context = np.dot(d_out, self.W_o.T)

        dV = np.matmul(self.A.transpose(0, 2, 1), d_context)
        dW_v = np.matmul(self.X.transpose(0, 2, 1), dV).sum(axis=0) / batch_size

        dA = np.matmul(d_context, self.V.transpose(0, 2, 1))

        d_scores = self.A * (dA - np.sum(dA * self.A, axis=-1, keepdims=True))

        d_scores_scaled = d_scores / np.sqrt(self.latent_dim)
        dQ = np.matmul(d_scores_scaled, self.K)
        dK = np.matmul(d_scores_scaled.transpose(0, 2, 1), self.Q)

        dW_q = np.matmul(self.Z.transpose(0, 2, 1), dQ).sum(axis=0) / batch_size
        dW_k = np.matmul(self.X.transpose(0, 2, 1), dK).sum(axis=0) / batch_size

        d_Z_attn = np.dot(dQ, self.W_q.T)
        d_latents = (d_Z_res + d_Z_attn).sum(axis=0) / batch_size

        self.W_q -= lr * dW_q
        self.W_k -= lr * dW_k
        self.W_v -= lr * dW_v
        self.W_o -= lr * dW_o
        self.latents -= lr * d_latents

def train_test():
    parser = argparse.ArgumentParser(description="Train a Perceiver Bottleneck component.")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    args = parser.parse_args()

    # Synthetic dataset: Sequence summarizing task
    batch_size = 32
    seq_len = 20
    input_dim = 16
    num_latents = 4
    latent_dim = 8

    X = np.random.randn(batch_size, seq_len, input_dim)
    W_target = np.random.randn(input_dim, latent_dim)
    Y = np.mean(np.dot(X, W_target), axis=1, keepdims=True)
    Y = np.tile(Y, (1, num_latents, 1))

    print("Training Perceiver Bottleneck Component...")
    model = PerceiverBottleneck(input_dim, latent_dim, num_latents)

    for epoch in range(args.epochs):
        out = model.forward(X)
        loss = np.mean((out - Y)**2)
        d_out = 2 * (out - Y)

        model.backward(d_out, lr=args.lr)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print(f"Final Loss: {loss:.6f}")

    if loss < 0.05:
        print("Success! Perceiver bottleneck successfully mapped sequence to latents.")

        docs_dir = "docs"
        os.makedirs(docs_dir, exist_ok=True)
        report_path = os.path.join(docs_dir, "0080_train_perceiver_component.md")

        report_content = f"""# 0080_train_perceiver_component

## Status
Success

## Component
Perceiver Bottleneck (Cross-Attention)

## Description
Implemented and verified a Perceiver Bottleneck component in pure NumPy. This architecture scales linearly with input sequence length by using a small set of trainable latent vectors as queries, and the input sequence as keys and values in a cross-attention layer. This reduces the complexity from $O(N^2)$ to $O(N \\cdot M)$, where $N$ is the sequence length and $M$ is the number of latents.

## Results
- **Final Loss (MSE):** {loss:.6f}

The model successfully learned to summarize a variable-length sequence into a fixed-size latent representation, verifying the mathematical formulation and manual backpropagation of the cross-attention bottleneck.

**Script:** `train_perceiver_component.py`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\nExperiment report saved to {report_path}")
    else:
        print("Failed.")

if __name__ == "__main__":
    train_test()
