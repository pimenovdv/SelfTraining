import numpy as np
import os
import argparse

np.random.seed(42)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class DirectFeedbackAlignmentMLP:
    """
    Multi-Layer Perceptron trained using Direct Feedback Alignment (DFA).
    Unlike standard backprop which propagates errors layer-by-layer backwards,
    DFA propagates the output error directly to each hidden layer using
    fixed random matrices. This allows for fully parallel weight updates
    across all hidden layers and is more biologically plausible.
    """
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim

        # Forward weights
        self.W1 = np.random.randn(input_dim, hidden1_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden1_dim))

        self.W2 = np.random.randn(hidden1_dim, hidden2_dim) * np.sqrt(2. / hidden1_dim)
        self.b2 = np.zeros((1, hidden2_dim))

        self.W3 = np.random.randn(hidden2_dim, output_dim) * np.sqrt(2. / hidden2_dim)
        self.b3 = np.zeros((1, output_dim))

        # Fixed random matrices to project output error directly to hidden layers
        self.B1 = np.random.randn(output_dim, hidden1_dim) * 0.1
        self.B2 = np.random.randn(output_dim, hidden2_dim) * 0.1

    def forward(self, X):
        self.X = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.h1 = relu(self.z1)

        self.z2 = np.dot(self.h1, self.W2) + self.b2
        self.h2 = relu(self.z2)

        self.z3 = np.dot(self.h2, self.W3) + self.b3
        self.out = self.z3
        return self.out

    def backward(self, d_out, lr=0.01):
        batch_size = d_out.shape[0]

        # Gradients for output layer
        dW3 = np.dot(self.h2.T, d_out) / batch_size
        db3 = np.sum(d_out, axis=0, keepdims=True) / batch_size

        # Direct Feedback Alignment for hidden layer 2
        dh2 = np.dot(d_out, self.B2)
        dz2 = dh2 * relu_derivative(self.z2)
        dW2 = np.dot(self.h1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        # Direct Feedback Alignment for hidden layer 1
        dh1 = np.dot(d_out, self.B1)
        dz1 = dh1 * relu_derivative(self.z1)
        dW1 = np.dot(self.X.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        # Updates
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

def train_test():
    parser = argparse.ArgumentParser(description="Train a MLP with Direct Feedback Alignment.")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    args = parser.parse_args()

    # Simple non-linear dataset (circle inside square)
    X = np.random.uniform(-1, 1, (400, 2))
    Y = np.zeros((400, 1))
    for i in range(400):
        if X[i, 0]**2 + X[i, 1]**2 < 0.5:
            Y[i, 0] = 1.0
        else:
            Y[i, 0] = 0.0

    print("Training MLP using Direct Feedback Alignment...")
    model = DirectFeedbackAlignmentMLP(2, 64, 64, 1)

    for epoch in range(args.epochs):
        out = model.forward(X)
        loss = np.mean((out - Y)**2)
        d_out = 2 * (out - Y)

        model.backward(d_out, lr=args.lr)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print(f"Final Loss: {loss:.6f}")

    if loss < 0.15:
        print("Success! Model learned using Direct Feedback Alignment.")

        docs_dir = "docs"
        os.makedirs(docs_dir, exist_ok=True)
        report_path = os.path.join(docs_dir, "0070_train_dfa_component.md")

        report_content = f"""# 0070_train_dfa_component

## Status
Success

## Component
Direct Feedback Alignment (DFA)

## Description
Implemented and trained a Multi-Layer Perceptron (MLP) using Direct Feedback Alignment (DFA) in pure NumPy. This explores biologically plausible learning rules by propagating the output error directly to each hidden layer using fixed random matrices, bypassing the backward pass through subsequent hidden layers entirely. This allows for parallel weight updates across layers.

## Results
- **Final Loss (MSE):** {loss:.6f}

The model successfully learned non-linear boundaries, confirming the hypothesis that directly projecting output errors via random matrices to hidden layers can provide a sufficient learning signal for intermediate representations to adapt.

**Script:** `train_dfa_component.py`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\nExperiment report saved to {report_path}")
    else:
        print("Failed.")

if __name__ == "__main__":
    train_test()
