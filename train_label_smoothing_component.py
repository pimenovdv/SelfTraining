import numpy as np
import argparse

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def label_smoothing_loss(logits, targets, smoothing=0.1):
    num_classes = logits.shape[-1]
    smoothed_targets = targets * (1 - smoothing) + smoothing / num_classes
    probs = softmax(logits)
    loss = -np.sum(smoothed_targets * np.log(np.clip(probs, 1e-7, 1.0)), axis=-1)
    return np.mean(loss)

def label_smoothing_grad(logits, targets, smoothing=0.1):
    num_classes = logits.shape[-1]
    smoothed_targets = targets * (1 - smoothing) + smoothing / num_classes
    probs = softmax(logits)
    return (probs - smoothed_targets) / logits.shape[0]

def main():
    parser = argparse.ArgumentParser(description="Train Label Smoothing Component")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--smoothing", type=float, default=0.1)
    args = parser.parse_args()

    np.random.seed(42)
    X = np.random.randn(100, 10)
    y_idx = np.random.randint(0, 5, size=(100,))
    y = np.zeros((100, 5))
    y[np.arange(100), y_idx] = 1.0

    W = np.random.randn(10, 5) * 0.1
    b = np.zeros(5)

    for epoch in range(args.epochs):
        logits = np.dot(X, W) + b
        loss = label_smoothing_loss(logits, y, smoothing=args.smoothing)

        grad_logits = label_smoothing_grad(logits, y, smoothing=args.smoothing)

        dW = np.dot(X.T, grad_logits)
        db = np.sum(grad_logits, axis=0)

        W -= args.lr * dW
        b -= args.lr * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    print("Label Smoothing Component Training Completed.")

if __name__ == "__main__":
    main()
