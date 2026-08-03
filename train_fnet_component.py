import numpy as np
import os
import argparse

class FNetBlock:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(hidden_dim, hidden_dim * 4) * 0.02
        self.b1 = np.zeros((1, 1, hidden_dim * 4))
        self.W2 = np.random.randn(hidden_dim * 4, hidden_dim) * 0.02
        self.b2 = np.zeros((1, 1, hidden_dim))

    def layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        out = gamma * (x - mean) / std + beta
        return out, mean, var, std

    def backward_layer_norm(self, dout, x, mean, var, std, gamma, eps=1e-5):
        N, S, D = x.shape
        x_mu = x - mean
        std_inv = 1. / std

        dx_norm = dout * gamma
        dvar = np.sum(dx_norm * x_mu, axis=-1, keepdims=True) * -.5 * std_inv**3
        dmean = np.sum(dx_norm * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2. * x_mu, axis=-1, keepdims=True)

        dx = dx_norm * std_inv + dvar * 2 * x_mu / D + dmean / D
        dgamma = np.sum(dout * x_mu * std_inv, axis=(0, 1), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 1), keepdims=True)

        return dx, dgamma, dbeta

    def forward(self, x):
        self.x = x
        self.fft_out = np.fft.fft2(x, axes=(1, 2))
        self.mixed = np.real(self.fft_out) / np.sqrt(x.shape[1] * x.shape[2])

        self.out1 = x + self.mixed

        self.gamma1 = np.ones((1, 1, self.hidden_dim))
        self.beta1 = np.zeros((1, 1, self.hidden_dim))
        self.norm1, self.mean1, self.var1, self.std1 = self.layer_norm(self.out1, self.gamma1, self.beta1)

        self.ffn_in = self.norm1
        self.ffn_hid = np.maximum(0, np.dot(self.ffn_in, self.W1) + self.b1)
        self.ffn_out = np.dot(self.ffn_hid, self.W2) + self.b2

        self.out2 = self.out1 + self.ffn_out

        self.gamma2 = np.ones((1, 1, self.hidden_dim))
        self.beta2 = np.zeros((1, 1, self.hidden_dim))
        self.norm2, self.mean2, self.var2, self.std2 = self.layer_norm(self.out2, self.gamma2, self.beta2)

        return self.norm2

    def backward(self, d_out2_norm, lr):
        d_out2, dgamma2, dbeta2 = self.backward_layer_norm(d_out2_norm, self.out2, self.mean2, self.var2, self.std2, self.gamma2)

        d_out1 = d_out2.copy()
        d_ffn_out = d_out2.copy()

        d_ffn_hid = np.dot(d_ffn_out, self.W2.T)
        dW2 = np.tensordot(self.ffn_hid, d_ffn_out, axes=([0, 1], [0, 1]))
        db2 = np.sum(d_ffn_out, axis=(0, 1), keepdims=True)

        d_ffn_in_pre = d_ffn_hid * (self.ffn_hid > 0)
        dW1 = np.tensordot(self.ffn_in, d_ffn_in_pre, axes=([0, 1], [0, 1]))
        db1 = np.sum(d_ffn_in_pre, axis=(0, 1), keepdims=True)

        d_ffn_in = np.dot(d_ffn_in_pre, self.W1.T)

        d_norm1 = d_ffn_in
        dx_out1, dgamma1, dbeta1 = self.backward_layer_norm(d_norm1, self.out1, self.mean1, self.var1, self.std1, self.gamma1)
        d_out1 += dx_out1

        dx = d_out1.copy()
        d_mixed = d_out1.copy()

        dx_mixed = np.real(np.fft.fft2(d_mixed, axes=(1, 2))) / np.sqrt(dx.shape[1] * dx.shape[2])
        dx += dx_mixed

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return dx

def train_test():
    parser = argparse.ArgumentParser(description="Train an FNet component.")
    args = parser.parse_args()

    np.random.seed(42)
    X = np.random.randn(8, 16, 32)
    Y = np.flip(X, axis=1)

    print("Training FNet component...")
    model = FNetBlock(32)

    for epoch in range(2500):
        out = model.forward(X)
        loss = np.mean((out - Y)**2)
        d_out = 2 * (out - Y) / (8 * 16 * 32)
        model.backward(d_out, 0.1)

    loss = np.mean((out - Y)**2)
    print(f"Final Loss: {loss:.6f}")

    if loss < 0.6:
        print("Success! Model learned sequence relationships using FNet block.")

        docs_dir = "docs"
        os.makedirs(docs_dir, exist_ok=True)
        report_path = os.path.join(docs_dir, "0081_train_fnet_component.md")

        report_content = f"""# 0081_train_fnet_component

## Status
Success

## Component
FNet Block

## Description
Implemented and evaluated an FNet block component using pure NumPy. This component tests the hypothesis that standard self-attention can be replaced by a parameter-free 2D Fourier Transform (mixing over sequence and hidden dimensions) while maintaining sequence modeling capabilities.

## Results
- **Final Loss (MSE):** {loss:.6f}

The model successfully learned sequence relationships (sequence inversion task) using the FNet block.

**Script:** `train_fnet_component.py`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\nExperiment report saved to {report_path}")
    else:
        print("Failed.")

if __name__ == "__main__":
    train_test()
