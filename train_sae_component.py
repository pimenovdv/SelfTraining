import numpy as np
import os
import argparse

np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder (SAE) component")
    parser.add_argument("--d_model", type=int, default=16, help="Input dimension")
    parser.add_argument("--d_hidden", type=int, default=64, help="Hidden dimension (expansion)")
    parser.add_argument("--l1_coeff", type=float, default=0.1, help="L1 regularization coefficient")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    return parser.parse_args()

def generate_data(num_samples=1000, d_model=16, d_hidden=64):
    # Generate sparse synthetic data
    # Real data is a combination of a few active features
    ground_truth_features = np.random.randn(d_hidden, d_model)
    data = np.zeros((num_samples, d_model))
    for i in range(num_samples):
        # Only 2-3 features active per sample
        active_indices = np.random.choice(d_hidden, size=3, replace=False)
        activations = np.random.rand(3) * 2.0
        data[i] = np.sum(ground_truth_features[active_indices] * activations[:, None], axis=0)
    return data

class SparseAutoencoder:
    def __init__(self, d_model, d_hidden, l1_coeff):
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        # Initialize weights
        self.W_e = np.random.randn(d_model, d_hidden) * np.sqrt(2.0 / d_model)
        self.b_enc = np.zeros((1, d_hidden))
        self.W_d = np.random.randn(d_hidden, d_model) * np.sqrt(2.0 / d_hidden)
        self.b_dec = np.zeros((1, d_model))

        # AdamW optimizer state
        self.mW_e, self.vW_e = np.zeros_like(self.W_e), np.zeros_like(self.W_e)
        self.mb_enc, self.vb_enc = np.zeros_like(self.b_enc), np.zeros_like(self.b_enc)
        self.mW_d, self.vW_d = np.zeros_like(self.W_d), np.zeros_like(self.W_d)
        self.mb_dec, self.vb_dec = np.zeros_like(self.b_dec), np.zeros_like(self.b_dec)
        self.step = 0

    def forward(self, x):
        self.x = x
        self.z_pre = np.dot(x, self.W_e) + self.b_enc
        self.z = np.maximum(0, self.z_pre) # ReLU
        self.x_hat = np.dot(self.z, self.W_d) + self.b_dec
        return self.x_hat, self.z

    def backward(self, lr):
        B = self.x.shape[0]

        # Gradients for Reconstruction Loss (MSE)
        # Loss = (1 / (B * d_model)) * sum((x_hat - x)^2)
        d_x_hat = 2.0 * (self.x_hat - self.x) / (B * self.d_model)

        dW_d = np.dot(self.z.T, d_x_hat)
        db_dec = np.sum(d_x_hat, axis=0, keepdims=True)

        d_z = np.dot(d_x_hat, self.W_d.T)

        # Gradients for L1 Loss
        # Loss_L1 = l1_coeff * (1 / B) * sum(|z|)
        # d_L1/d_z = l1_coeff / B * sign(z) (and since z >= 0 after ReLU, sign(z) = 1 for z > 0)
        d_z_l1 = (self.l1_coeff / B) * (self.z > 0)
        d_z += d_z_l1

        d_z_pre = d_z * (self.z_pre > 0)

        dW_e = np.dot(self.x.T, d_z_pre)
        db_enc = np.sum(d_z_pre, axis=0, keepdims=True)

        # Update weights using Adam
        self.step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def adam_update(W, dW, mW, vW):
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * (dW ** 2)
            m_hat = mW / (1 - beta1**self.step)
            v_hat = vW / (1 - beta2**self.step)
            W -= lr * m_hat / (np.sqrt(v_hat) + eps)
            return W, mW, vW

        self.W_d, self.mW_d, self.vW_d = adam_update(self.W_d, dW_d, self.mW_d, self.vW_d)
        self.b_dec, self.mb_dec, self.vb_dec = adam_update(self.b_dec, db_dec, self.mb_dec, self.vb_dec)
        self.W_e, self.mW_e, self.vW_e = adam_update(self.W_e, dW_e, self.mW_e, self.vW_e)
        self.b_enc, self.mb_enc, self.vb_enc = adam_update(self.b_enc, db_enc, self.mb_enc, self.vb_enc)

def main():
    args = parse_args()

    dataset = generate_data(num_samples=1000, d_model=args.d_model, d_hidden=args.d_hidden)
    model = SparseAutoencoder(args.d_model, args.d_hidden, args.l1_coeff)

    batch_size = 64

    for epoch in range(args.epochs):
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        x = dataset[indices]

        x_hat, z = model.forward(x)

        mse_loss = np.mean((x_hat - x)**2)
        l1_loss = args.l1_coeff * np.mean(np.sum(np.abs(z), axis=1))
        loss = mse_loss + l1_loss

        model.backward(lr=args.lr)

        if epoch % 1000 == 0:
            active_neurons = np.mean(np.sum(z > 0, axis=1))
            print(f"Epoch {epoch}, Loss: {loss:.4f} (MSE: {mse_loss:.4f}, L1: {l1_loss:.4f}), Active: {active_neurons:.1f}")

    final_loss = loss
    success = mse_loss < 0.1 and active_neurons < args.d_hidden * 0.2

    print(f"Final Loss: {final_loss:.4f}")

    os.makedirs("docs", exist_ok=True)
    doc_path = "docs/0039_train_sae_component.md"

    doc_content = f"""# Experiment 0039: Sparse Autoencoder (SAE) Component

## Objective
Implement and verify a Sparse Autoencoder (SAE) using pure NumPy. The goal is to mathematically model an autoencoder that learns a sparse, overcomplete representation of the data, which is commonly used in mechanistic interpretability.

## Mathematical Formulation

### Forward Pass
Let $x \\in \\mathbb{{R}}^D$ be the input data.
The encoder maps the input to a higher-dimensional hidden space $F$ ($F > D$) with a ReLU activation to encourage non-negativity:
$z = \\text{{ReLU}}(x W_e + b_{{enc}})$
The decoder attempts to reconstruct the original input:
$\\hat{{x}} = z W_d + b_{{dec}}$

### Loss Function
The model is trained to minimize the reconstruction error (Mean Squared Error) while encouraging sparsity in the latent representation via an L1 penalty:
$\\mathcal{{L}} = \\frac{{1}}{{B \\cdot D}} \\sum_{{i,j}} (x_{{i,j}} - \\hat{{x}}_{{i,j}})^2 + \\lambda \\frac{{1}}{{B}} \\sum_{{i,k}} |z_{{i,k}}|$

### Backward Pass
Gradients are calculated manually:
$\\frac{{\\partial \\mathcal{{L}}_{{MSE}}}}{{\\partial \\hat{{x}}}} = \\frac{{2}}{{B \\cdot D}} (\\hat{{x}} - x)$
$\\frac{{\\partial \\mathcal{{L}}_{{L1}}}}{{\\partial z}} = \\frac{{\\lambda}}{{B}} \\text{{sign}}(z)$
These are routed back through the decoder and encoder via the chain rule.

## Results
- **Status:** {"Success" if success else "Failed"}
- **Final Total Loss:** {final_loss:.4f}
- **Final MSE Loss:** {mse_loss:.4f}
- **Final L1 Loss:** {l1_loss:.4f}
- **Average Active Neurons:** {active_neurons:.1f} / {args.d_hidden}
- **Epochs:** {args.epochs}

## Conclusion
{"The model successfully learned to reconstruct the input data while maintaining a sparse latent representation, verifying the mathematical soundness of the SAE formulation and its manual backpropagation." if success else "The model failed to converge adequately or achieve the desired sparsity. Further tuning is required."}
"""

    with open(doc_path, "w") as f:
        f.write(doc_content)

    print(f"Documentation saved to {doc_path}")

    if not success:
        exit(1)

if __name__ == "__main__":
    main()
