import numpy as np

class PredictiveCodingNetwork:
    """
    Predictive Coding Network (PCN)

    This component tests a biologically plausible alternative to backpropagation
    using local learning rules and iterative inference.

    Instead of calculating gradients via the chain rule, a PCN maintains a
    state (value nodes) and errors (error nodes) at each layer.
    During an inference phase, it minimizes the prediction error by updating
    the state nodes. During a weight update phase, it updates the weights
    using a simple Hebbian-like local rule based on the errors.
    """
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # Initialize weights and biases
        # W[i] projects from layer i to layer i+1
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2.0/layer_sizes[i-1])
                  for i in range(1, self.num_layers)]
        self.b = [np.zeros((layer_sizes[i], 1)) for i in range(1, self.num_layers)]

    def act(self, x):
        """Activation function (Tanh)"""
        return np.tanh(x)

    def act_deriv(self, x):
        """Derivative of the activation function"""
        return 1.0 - np.tanh(x)**2

    def train_step(self, x, y, inference_steps=20, inference_lr=0.1, weight_lr=0.01):
        """
        Performs one training step (inference phase + weight update phase).
        """
        # 1. Initialization Phase
        v = [np.zeros((self.W[i].shape[0], 1)) for i in range(len(self.W))]
        v.insert(0, x) # v[0] is clamped to input x

        for i in range(1, self.num_layers - 1):
            v[i] = self.W[i-1] @ self.act(v[i-1]) + self.b[i-1]

        # Clamp the output layer to target y
        v[-1] = y

        # 2. Inference Phase (Iterative settling)
        for step in range(inference_steps):
            errors = []
            for i in range(1, self.num_layers):
                mu_i = self.W[i-1] @ self.act(v[i-1]) + self.b[i-1]
                errors.append(v[i] - mu_i)

            for i in range(1, self.num_layers - 1):
                e_i = errors[i-1]
                e_next = errors[i]

                grad_v_i = e_i - (self.W[i].T @ e_next) * self.act_deriv(v[i])
                v[i] -= inference_lr * grad_v_i

        # 3. Weight Update Phase (Local learning)
        errors = []
        for i in range(1, self.num_layers):
            mu_i = self.W[i-1] @ self.act(v[i-1]) + self.b[i-1]
            errors.append(v[i] - mu_i)

        for i in range(self.num_layers - 1):
            e_i = errors[i]

            grad_W = -e_i @ self.act(v[i]).T
            grad_b = -e_i

            self.W[i] -= weight_lr * grad_W
            self.b[i] -= weight_lr * np.sum(grad_b, axis=1, keepdims=True)

        return sum(np.sum(e**2) for e in errors)

    def predict(self, x):
        """
        Forward pass for prediction (without iterative inference).
        """
        v = x
        for i in range(self.num_layers - 1):
            v = self.W[i] @ self.act(v) + self.b[i]
        return v

def test_pcn():
    print("Testing Predictive Coding Network (PCN) Component...")

    np.random.seed(42)

    print("Generating dataset (Sine Wave Regression)...")
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    Y = np.sin(X)

    layer_sizes = [1, 16, 16, 1]
    pcn = PredictiveCodingNetwork(layer_sizes)

    epochs = 100
    inference_steps = 30
    inference_lr = 0.05
    weight_lr = 0.01

    print(f"Training PCN with architecture {layer_sizes} for {epochs} epochs...")

    for epoch in range(epochs):
        total_energy = 0
        indices = np.random.permutation(len(X))

        for idx in indices:
            x_i = X[idx].reshape(-1, 1)
            y_i = Y[idx].reshape(-1, 1)

            energy = pcn.train_step(x_i, y_i,
                                    inference_steps=inference_steps,
                                    inference_lr=inference_lr,
                                    weight_lr=weight_lr)
            total_energy += energy

        if (epoch + 1) % 20 == 0 or epoch == 0:
            pred_y = np.array([pcn.predict(x.reshape(-1, 1))[0,0] for x in X])
            mse = np.mean((pred_y - Y.flatten())**2)
            avg_energy = total_energy / len(X)
            print(f"Epoch {epoch+1:3d}/{epochs} | Avg Energy: {avg_energy:.4f} | MSE: {mse:.4f}")

    pred_y = np.array([pcn.predict(x.reshape(-1, 1))[0,0] for x in X])
    final_mse = np.mean((pred_y - Y.flatten())**2)

    print("\n--- Final Results ---")
    print(f"Final Mean Squared Error: {final_mse:.4f}")

    if final_mse < 0.05:
        print("Success! The Predictive Coding Network successfully learned the non-linear function using local learning rules.")

        doc_content = f"""# Component: Predictive Coding Network (PCN)

**Script:** `train_pcn_component.py`

## Description
This component evaluates a **Predictive Coding Network (PCN)** using pure NumPy. Predictive Coding is a biologically plausible alternative to backpropagation that relies on local learning rules and an iterative inference phase, rather than a global backward pass.

In a PCN, the network maintains both state nodes (values) and error nodes at each layer. The learning process consists of two phases:
1.  **Inference Phase:** The input and target output are clamped. The hidden state nodes are iteratively updated via gradient descent to minimize the local prediction errors (the difference between the state node's value and the top-down prediction from the previous layer).
2.  **Weight Update Phase:** Once the states have settled (or after a fixed number of steps), the weights and biases are updated using a local, Hebbian-like learning rule based solely on the pre-synaptic activations and the post-synaptic errors.

This approach avoids the weight transport problem and non-local credit assignment issues of standard backpropagation, offering insights into how biological brains might perform credit assignment.

## Mathematical Formulation
Let $v_i$ be the state nodes at layer $i$, and $W_i, b_i$ be the weights and biases connecting layer $i$ to $i+1$.
The top-down prediction for layer $i$ is:
$$\\mu_i = W_{{i-1}} \\sigma(v_{{i-1}}) + b_{{i-1}}$$

The local prediction error at layer $i$ is:
$$e_i = v_i - \\mu_i$$

The total network energy is the sum of squared errors:
$$E = \\frac{{1}}{{2}} \\sum_i ||e_i||^2$$

**Inference Phase (Updating $v_i$):**
$$\\Delta v_i \\propto -\\frac{{\\partial E}}{{\\partial v_i}} = -e_i + (W_i^T e_{{i+1}}) \\odot \\sigma'(v_i)$$

**Weight Update Phase (Updating $W_i$):**
$$\\Delta W_i \\propto -\\frac{{\\partial E}}{{\\partial W_i}} = e_{{i+1}} \\sigma(v_i)^T$$

## Experiment Results
*   **Task:** Non-linear regression (Sine Wave).
*   **Architecture:** [1, 16, 16, 1]
*   **Result:** The PCN successfully learned to approximate the sine wave, reducing the Mean Squared Error to {final_mse:.4f}.
*   **Observation:** The local inference-based learning rule was able to effectively train deep representations, demonstrating a viable, biologically motivated alternative to end-to-end backpropagation for representation learning.
"""
        import os
        docs_dir = "docs"
        os.makedirs(docs_dir, exist_ok=True)
        import glob
        existing_docs = glob.glob(os.path.join(docs_dir, "*_train_*.md"))
        next_num = len(existing_docs) + 1
        doc_filename = os.path.join(docs_dir, f"{next_num:04d}_train_pcn_component.md")

        with open(doc_filename, "w") as f:
            f.write(doc_content)
        print(f"Documentation saved to {doc_filename}")

    else:
        print("Failure. The PCN failed to converge to an acceptable error level.")

if __name__ == "__main__":
    test_pcn()
