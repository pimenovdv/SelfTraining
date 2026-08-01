import numpy as np
import os
import argparse

np.random.seed(42)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class FeedbackAlignmentMLP:
    """
    Multi-Layer Perceptron trained using Random Feedback Alignment (FA).
    Unlike standard backpropagation which uses the transpose of the forward
    weights (W^T) to propagate errors, FA uses a fixed random matrix (B).
    This avoids the "weight transport problem", making the learning rule
    more biologically plausible. The network learns because the forward
    weights adapt to make the random backward weights effective.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Forward weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # Random fixed feedback matrix B for the backward pass
        self.B = np.random.randn(hidden_dim, output_dim) * 0.1

    def forward(self, X):
        self.X = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.h1 = relu(self.z1)

        self.z2 = np.dot(self.h1, self.W2) + self.b2
        self.out = self.z2
        return self.out

    def backward(self, d_out, lr=0.01):
        batch_size = d_out.shape[0]

        # Gradients for output layer (standard)
        dW2 = np.dot(self.h1.T, d_out) / batch_size
        db2 = np.sum(d_out, axis=0, keepdims=True) / batch_size

        # Feedback Alignment: Use fixed random matrix B instead of W2.T
        # Standard backprop would be: dh1 = np.dot(d_out, self.W2.T)
        dh1 = np.dot(d_out, self.B.T)

        dz1 = dh1 * relu_derivative(self.z1)

        dW1 = np.dot(self.X.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        # Updates
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

def train_test():
    parser = argparse.ArgumentParser(description="Train a MLP with Random Feedback Alignment.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    args = parser.parse_args()

    # Simple XOR-like continuous dataset
    X = np.random.randn(100, 2)
    Y = np.zeros((100, 1))
    for i in range(100):
        if (X[i, 0] > 0 and X[i, 1] > 0) or (X[i, 0] < 0 and X[i, 1] < 0):
            Y[i, 0] = 1.0
        else:
            Y[i, 0] = 0.0

    print("Training MLP using Random Feedback Alignment...")
    model = FeedbackAlignmentMLP(2, 32, 1)

    for epoch in range(args.epochs):
        out = model.forward(X)
        loss = np.mean((out - Y)**2)
        d_out = 2 * (out - Y)

        model.backward(d_out, lr=args.lr)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print(f"Final Loss: {loss:.6f}")

    if loss < 0.1:
        print("Success! Model learned using Feedback Alignment.")

        docs_dir = "docs"
        os.makedirs(docs_dir, exist_ok=True)
        report_path = os.path.join(docs_dir, "0069_train_feedback_alignment_component.md")

        report_content = f"""# 0069_train_feedback_alignment_component

## Status
Success

## Component
Random Feedback Alignment (FA)

## Description
Implemented and trained a Multi-Layer Perceptron (MLP) using Random Feedback Alignment (FA) in pure NumPy. This explores biologically plausible learning rules by avoiding the 'weight transport problem' inherent in standard backpropagation. Instead of using the transpose of the forward weights ($W^T$) to propagate errors, FA uses a fixed random weight matrix ($B$).

## Results
- **Final Loss (MSE):** {loss:.6f}

The model successfully learned non-linear boundaries (XOR-like data), confirming the hypothesis that gradients propagated through fixed random matrices can still provide a useful learning signal, as the forward weights adapt to align with the random backward weights.

**Script:** `train_feedback_alignment_component.py`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\\nExperiment report saved to {report_path}")
    else:
        print("Failed.")

if __name__ == "__main__":
    train_test()
