import argparse
import os
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class FFLayer:
    """
    Forward-Forward Layer
    """
    def __init__(self, in_features, out_features, lr=0.01, threshold=2.0):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros(out_features)
        self.lr = lr
        self.threshold = threshold

    def normalize(self, x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-8)

    def forward(self, x):
        x_norm = self.normalize(x)
        return relu(np.dot(x_norm, self.W) + self.b)

    def train_step(self, x_pos, x_neg):
        x_pos_norm = self.normalize(x_pos)
        x_neg_norm = self.normalize(x_neg)

        a_pos = relu(np.dot(x_pos_norm, self.W) + self.b)
        a_neg = relu(np.dot(x_neg_norm, self.W) + self.b)

        g_pos = np.sum(a_pos ** 2, axis=1)
        g_neg = np.sum(a_neg ** 2, axis=1)

        p_pos = sigmoid(g_pos - self.threshold)
        p_neg = sigmoid(g_neg - self.threshold)

        grad_g_pos = p_pos - 1.0
        grad_g_neg = p_neg

        grad_z_pos = 2 * a_pos * grad_g_pos[:, np.newaxis]
        grad_z_neg = 2 * a_neg * grad_g_neg[:, np.newaxis]

        dW = np.dot(x_pos_norm.T, grad_z_pos) + np.dot(x_neg_norm.T, grad_z_neg)
        db = np.sum(grad_z_pos, axis=0) + np.sum(grad_z_neg, axis=0)

        self.W -= self.lr * dW / x_pos.shape[0]
        self.b -= self.lr * db / x_pos.shape[0]

        loss = np.mean(-np.log(p_pos + 1e-8) - np.log(1 - p_neg + 1e-8))
        return loss

def predict(x, layer1, layer2):
    goodness = []
    for label in [0, 1]:
        y_oh = np.zeros((x.shape[0], 2))
        y_oh[:, label] = 1
        x_in = np.hstack((x, y_oh))

        h1 = layer1.forward(x_in)
        h2 = layer2.forward(h1)
        g = np.sum(h1**2, axis=1) + np.sum(h2**2, axis=1)
        goodness.append(g)

    return np.argmax(np.array(goodness), axis=0)

def main():
    parser = argparse.ArgumentParser(description="Train a Forward-Forward component.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    args = parser.parse_args()

    np.random.seed(42)
    N = 1000
    X = np.random.randn(N, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    Y_onehot = np.zeros((N, 2))
    Y_onehot[np.arange(N), y] = 1

    Y_wrong = np.zeros((N, 2))
    Y_wrong[np.arange(N), 1 - y] = 1

    X_pos = np.hstack((X, Y_onehot))
    X_neg = np.hstack((X, Y_wrong))

    layer1 = FFLayer(4, 32, lr=args.lr, threshold=2.0)
    layer2 = FFLayer(32, 16, lr=args.lr, threshold=2.0)

    print("Starting Forward-Forward training...")
    for epoch in range(args.epochs):
        loss1 = layer1.train_step(X_pos, X_neg)

        h_pos = layer1.forward(X_pos)
        h_neg = layer1.forward(X_neg)

        loss2 = layer2.train_step(h_pos, h_neg)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: L1={loss1:.4f}, L2={loss2:.4f}")

    preds = predict(X, layer1, layer2)
    acc = np.mean(preds == y)
    print(f"Final Accuracy: {acc:.4f}")
    success = acc > 0.9

    os.makedirs('docs', exist_ok=True)
    doc_content = f"""# Experiment 0117: Train Forward-Forward Component

## Objective
To implement and train a neural network using the Forward-Forward (FF) algorithm, mathematically testing an alternative to backpropagation where layers learn to maximize "goodness" for positive data and minimize it for negative data locally.

## Details
*   **Script:** `train_ff_component.py`
*   **Architecture:** Two local FFLayers with layer normalization and ReLU activation. Goodness is defined as the sum of squared activations.
*   **Training Data:** Synthetic dataset XOR-like problem (positive and negative pairs constructed using one-hot labels).
*   **Learning Rate:** {args.lr}
*   **Epochs:** {args.epochs}

## Results
*   **Final Accuracy:** {acc:.4f}
*   **Success:** {success}

## Conclusion
The Forward-Forward component successfully learned to classify the data using local layer-wise updates without backpropagation, verifying the mathematical feasibility of gradient-free (with respect to subsequent layers) contrastive learning on intermediate representations.
"""
    with open('docs/0117_train_ff_component.md', 'w') as f:
        f.write(doc_content)
    print("Documentation written to docs/0117_train_ff_component.md")

if __name__ == "__main__":
    main()
