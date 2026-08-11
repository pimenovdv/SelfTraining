import numpy as np
import os
import time

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): s = sigmoid(x); return s * (1 - s)
def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
def softplus_deriv(x): return sigmoid(x)

class PositionalEncoding:
    """Positional encoding for spatial coordinates as used in NeRF."""
    def __init__(self, L=10):
        self.L = L

    def encode(self, x):
        encoded = [x]
        for i in range(self.L):
            encoded.append(np.sin((2.0 ** i) * np.pi * x))
            encoded.append(np.cos((2.0 ** i) * np.pi * x))
        return np.concatenate(encoded, axis=-1)

class NeRFMLP:
    """A simplified Multilayer Perceptron for Neural Radiance Fields."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        # Initialize weights with He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b3 = np.zeros(output_dim)

    def forward(self, x):
        self.x = x
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = relu(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3

        # Density (sigma) uses softplus to ensure non-negativity
        # RGB uses sigmoid to ensure values are in [0, 1]
        self.sigma = softplus(self.z3[:, 0:1])
        self.rgb = sigmoid(self.z3[:, 1:4])
        return self.sigma, self.rgb

    def backward(self, d_sigma, d_rgb):
        # Gradients of outputs
        d_z3_sigma = d_sigma * softplus_deriv(self.z3[:, 0:1])
        d_z3_rgb = d_rgb * sigmoid_deriv(self.z3[:, 1:4])
        d_z3 = np.concatenate([d_z3_sigma, d_z3_rgb], axis=1)

        # Layer 3
        self.dW3 = np.dot(self.a2.T, d_z3)
        self.db3 = np.sum(d_z3, axis=0)

        # Layer 2
        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2 * relu_deriv(self.z2)

        self.dW2 = np.dot(self.a1.T, d_z2)
        self.db2 = np.sum(d_z2, axis=0)

        # Layer 1
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * relu_deriv(self.z1)

        self.dW1 = np.dot(self.x.T, d_z1)
        self.db1 = np.sum(d_z1, axis=0)

    def update(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3

def volume_render_forward(sigma, rgb, deltas):
    """
    Computes volumetric rendering along a ray.
    sigma: (N, 1) volume density
    rgb: (N, 3) color
    deltas: (N, 1) distance between adjacent samples
    """
    # Probability of terminating at the current point (alpha)
    alpha = 1.0 - np.exp(-sigma * deltas)

    # Transmittance (T): probability of ray reaching this point
    T = np.ones_like(alpha)
    for i in range(1, len(alpha)):
        T[i] = T[i-1] * (1.0 - alpha[i-1] + 1e-10) # 1e-10 for numerical stability

    # Weights for each point's color contribution
    weights = T * alpha

    # Expected color
    C = np.sum(weights * rgb, axis=0)

    return C, alpha, T, weights

def volume_render_backward(d_C, C, alpha, T, weights, rgb, sigma, deltas):
    """
    Computes gradients of the volume rendering equations.
    """
    N = len(alpha)

    # Gradient of C w.r.t weights and rgb
    d_weights = np.sum(d_C * rgb, axis=1, keepdims=True)
    d_rgb = d_C * weights

    # Gradient of weights w.r.t alpha and T
    d_T = d_weights * alpha
    d_alpha = d_weights * T

    # Gradient of T w.r.t alpha
    d_alpha_from_T = np.zeros_like(alpha)
    for j in range(N):
        for i in range(j + 1, N):
            term = -T[i] / (1.0 - alpha[j] + 1e-10)
            d_alpha_from_T[j] += d_T[i] * term

    # Total gradient of alpha
    d_alpha_total = d_alpha + d_alpha_from_T

    # Gradient of alpha w.r.t sigma
    d_sigma = d_alpha_total * deltas * np.exp(-sigma * deltas)

    return d_sigma, d_rgb

def train_nerf_component():
    print("Starting Neural Radiance Field (NeRF) component training...")
    np.random.seed(42)

    # 1. Generate a synthetic ray with target color
    N_points = 64
    z_vals = np.linspace(0.1, 1.0, N_points)
    deltas = np.diff(z_vals, append=z_vals[-1] + (z_vals[1]-z_vals[0])).reshape(-1, 1)

    pts = np.zeros((N_points, 3))
    pts[:, 2] = z_vals

    target_C = np.array([0.9, 0.1, 0.5]) # Target pixel color (magenta-ish)

    # 2. Setup model
    pe = PositionalEncoding(L=4)
    encoded_pts = pe.encode(pts)
    input_dim = encoded_pts.shape[-1]

    nerf = NeRFMLP(input_dim=input_dim, hidden_dim=64, output_dim=4)

    epochs = 1500
    lr = 0.05
    losses = []

    start_time = time.time()

    print(f"Target Color: {target_C}")
    print("Training MLP to overfit a single ray...")
    for epoch in range(epochs):
        # Forward pass
        sigma, rgb = nerf.forward(encoded_pts)
        C, alpha, T, weights = volume_render_forward(sigma, rgb, deltas)

        # Loss
        loss = 0.5 * np.sum((C - target_C)**2)
        losses.append(loss)

        # Backward pass
        d_C = C - target_C
        d_sigma, d_rgb = volume_render_backward(d_C, C, alpha, T, weights, rgb, sigma, deltas)
        nerf.backward(d_sigma, d_rgb)

        # Update weights
        nerf.update(lr)

        if (epoch + 1) % 150 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss:.6f} | Rendered Color: {C.round(4)}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Validation
    final_loss = losses[-1]
    success = final_loss < 0.01

    print(f"\nFinal Loss: {final_loss:.6f}")
    if success:
        print("Success! NeRF model successfully learned to represent the ray and render the target color.")
    else:
        print("Failure. NeRF model did not converge to the target color.")

    # Generate documentation
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)

    # Find next sequence number
    existing_docs = [f for f in os.listdir(docs_dir) if f.endswith('.md')]
    max_num = 0
    for doc in existing_docs:
        try:
            num = int(doc.split('_')[0])
            max_num = max(max_num, num)
        except ValueError:
            pass
    next_num = max_num + 1

    doc_filename = f"{next_num:04d}_train_nerf_component.md"
    doc_path = os.path.join(docs_dir, doc_filename)

    with open(doc_path, "w") as f:
        f.write(f"# Neural Radiance Field (NeRF) Component Training\n\n")
        f.write(f"**Script:** `train_nerf_component.py`\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n")
        f.write(f"**Status:** {'Success' if success else 'Failure'}\n\n")
        f.write(f"## Mathematical Background\n\n")
        f.write(f"Neural Radiance Fields (NeRF) represent a continuous 3D scene as a function mapped by a neural network. ")
        f.write(f"The input is a 3D coordinate (often with viewing direction, omitted here for simplicity), and the output is volume density $\\sigma$ and color $c = (r, g, b)$.\n\n")
        f.write(f"Volume rendering along a ray is computed using the integral:\n")
        f.write(r"$$ C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) c(\mathbf{r}(t), \mathbf{d}) dt $$" + "\n\n")
        f.write(f"Where $T(t)$ is the accumulated transmittance:\n")
        f.write(r"$$ T(t) = \exp\left(-\int_{t_n}^t \sigma(\mathbf{r}(s)) ds\right) $$" + "\n\n")
        f.write(f"This script implements the discrete approximation using classical alpha compositing, and computes full gradients mathematically in pure NumPy.\n\n")
        f.write(f"## Experiment Details\n\n")
        f.write(f"- Modeled a single ray with 64 sample points.\n")
        f.write(f"- Network: MLP with 2 hidden layers (64 units), Softplus activation for density, Sigmoid for color.\n")
        f.write(f"- Target Color: {target_C}\n")
        f.write(f"- Final Rendered Color: {C.round(4)}\n")
        f.write(f"- Final Loss: {final_loss:.6f}\n")

    print(f"Documentation saved to {doc_path}")
    return success

if __name__ == "__main__":
    train_nerf_component()
