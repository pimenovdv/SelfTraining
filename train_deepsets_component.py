import numpy as np
import os

class MLP:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        activations = [x]
        zs = []
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = z # Linear output for the last layer
            activations.append(a)
        return activations, zs

    def backward(self, grad_output, activations, zs, lr):
        grad = grad_output
        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]

            if i < len(self.weights) - 1:
                grad = grad * self.relu_deriv(zs[i])

            # Compute gradients
            if len(grad.shape) == 3:
                # Batched sequence/set data (B, N, D)
                grad_w = np.sum(np.matmul(a_prev.transpose(0, 2, 1), grad), axis=0)
                grad_b = np.sum(grad, axis=(0, 1))
                grad_prev = np.matmul(grad, self.weights[i].T)
            else:
                # Standard batched data (B, D)
                grad_w = np.dot(a_prev.T, grad)
                grad_b = np.sum(grad, axis=0)
                grad_prev = np.dot(grad, self.weights[i].T)

            # Update weights
            self.weights[i] -= lr * grad_w
            self.biases[i] -= lr * grad_b

            grad = grad_prev

        return grad

class DeepSets:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Phi network applied to each element independently
        self.phi = MLP([input_dim, hidden_dim, hidden_dim])
        # Rho network applied to the aggregated representation
        self.rho = MLP([hidden_dim, hidden_dim, output_dim])

    def forward(self, x):
        # x shape: (batch_size, num_elements, input_dim)

        # Apply phi to each element
        # Flatten for MLP: (batch_size * num_elements, input_dim)
        b, n, d = x.shape
        x_flat = x.reshape(-1, d)

        phi_activations, phi_zs = self.phi.forward(x_flat)
        phi_out = phi_activations[-1].reshape(b, n, -1)

        # Aggregate (sum pooling for permutation invariance)
        agg = np.sum(phi_out, axis=1)

        # Apply rho to aggregated representation
        rho_activations, rho_zs = self.rho.forward(agg)
        out = rho_activations[-1]

        return out, phi_activations, phi_zs, rho_activations, rho_zs, phi_out, agg

    def backward(self, grad_output, x, phi_activations, phi_zs, rho_activations, rho_zs, phi_out, agg, lr):
        b, n, d = x.shape

        # Backward through rho
        grad_agg = self.rho.backward(grad_output, rho_activations, rho_zs, lr)

        # Gradient through sum pooling (broadcast to all elements)
        grad_phi_out = np.repeat(grad_agg[:, np.newaxis, :], n, axis=1)

        # Backward through phi
        grad_phi_out_flat = grad_phi_out.reshape(-1, grad_phi_out.shape[-1])
        self.phi.backward(grad_phi_out_flat, phi_activations, phi_zs, lr)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def generate_set_data(num_samples, max_set_size, input_dim):
    data = []
    labels = []

    for _ in range(num_samples):
        # Random set size
        set_size = np.random.randint(1, max_set_size + 1)

        # Generate random set elements
        elements = np.random.randn(set_size, input_dim)

        # Pad to max_set_size with zeros (masking usually needed for variable sizes,
        # but for simplicity we fix size in this basic NumPy test)
        # We will just generate fixed size sets to keep the batching simple for numpy

    # For simplicity, fixed size sets for batch processing
    sets = np.random.randn(num_samples, max_set_size, input_dim)

    # Target task: Is the sum of the first feature across all elements > 0?
    sums = np.sum(sets[:, :, 0], axis=1)
    labels = (sums > 0).astype(float).reshape(-1, 1)

    return sets, labels

def main():
    np.random.seed(42)

    # Hyperparameters
    batch_size = 32
    input_dim = 2
    hidden_dim = 16
    output_dim = 1
    lr = 0.01
    epochs = 200

    # Dataset
    X_train, y_train = generate_set_data(1000, max_set_size=5, input_dim=input_dim)

    model = DeepSets(input_dim, hidden_dim, output_dim)

    print("Training Deep Sets model on Set Sum Thresholding Task...")

    for epoch in range(epochs):
        # Mini-batching
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0
        correct = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            # Forward pass
            out, phi_a, phi_z, rho_a, rho_z, phi_out, agg = model.forward(X_batch)

            # Sigmoid activation for binary classification
            preds = sigmoid(out)

            # Binary Cross Entropy Loss
            loss = -np.mean(y_batch * np.log(preds + 1e-8) + (1 - y_batch) * np.log(1 - preds + 1e-8))
            epoch_loss += loss * len(X_batch)

            correct += np.sum((preds > 0.5) == y_batch)

            # Gradients (derivative of BCE + sigmoid)
            grad = (preds - y_batch) / len(X_batch)

            # Backward pass
            model.backward(grad, X_batch, phi_a, phi_z, rho_a, rho_z, phi_out, agg, lr)

        epoch_loss /= len(X_train)
        accuracy = correct / len(X_train)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f}")

    # Verify permutation invariance
    print("\nVerifying Permutation Invariance:")
    test_set = X_train[0:1] # Take first set
    permuted_set = test_set[:, np.random.permutation(test_set.shape[1]), :]

    out1, *_ = model.forward(test_set)
    out2, *_ = model.forward(permuted_set)

    diff = np.abs(out1 - out2).sum()
    print(f"Original output: {out1[0,0]:.6f}")
    print(f"Permuted output: {out2[0,0]:.6f}")
    print(f"Absolute difference: {diff:.6e}")

    success = accuracy > 0.9 and diff < 1e-5

    # Document results
    os.makedirs('docs', exist_ok=True)
    with open('docs/0084_train_deepsets_component.md', 'w') as f:
        f.write("# Experiment 0084: Deep Sets Component Training\n\n")
        f.write("**Script:** `train_deepsets_component.py`\n\n")
        f.write("## Objective\n")
        f.write("Evaluate a Deep Sets component utilizing element-wise $\\phi$ network and a symmetric aggregation function followed by a $\\rho$ network, ensuring permutation invariance.\n\n")
        f.write("## Methodology\n")
        f.write("- Implemented a `DeepSets` class with independent element-wise MLPs and a sum-pooling aggregator.\n")
        f.write("- Trained on a binary classification task to determine if the sum of a specific feature across set elements is positive.\n")
        f.write("- Verified permutation invariance by comparing outputs of a given set and its shuffled version.\n\n")
        f.write("## Results\n")
        f.write(f"- Final Accuracy: {accuracy:.4f}\n")
        f.write(f"- Output Difference on Permuted Set: {diff:.6e}\n")
        if success:
            f.write("- **Status**: SUCCESS\n")
            f.write("- The model successfully learned to process unordered sets and maintains strict permutation invariance.\n")
        else:
            f.write("- **Status**: FAILED\n")
            f.write("- The model did not achieve the required accuracy or permutation invariance.\n")

    print("\nDocumentation generated at docs/0084_train_deepsets_component.md")

if __name__ == "__main__":
    main()
