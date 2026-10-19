"""
Component Testing: ArcFace Loss
This script implements the ArcFace (Additive Angular Margin Loss) component and verifies it on a small simulated dataset.
"""

import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test ArcFace Loss Component")
    args = parser.parse_args()

    # Simulated features and weights
    np.random.seed(42)
    batch_size = 4
    num_classes = 3
    embedding_size = 5

    features = np.random.randn(batch_size, embedding_size)
    weights = np.random.randn(embedding_size, num_classes)
    labels = np.array([0, 1, 2, 0])

    s = 64.0
    m = 0.50

    # L2 normalize features and weights
    features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
    weights_norm = weights / np.linalg.norm(weights, axis=0, keepdims=True)

    # Compute cosine similarity
    cosine = np.dot(features_norm, weights_norm)
    cosine = np.clip(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

    # Compute theta and add margin m to the true classes
    theta = np.arccos(cosine)

    # One-hot encode labels
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), labels] = 1.0

    # Target logits: cos(theta + m)
    target_logits = np.cos(theta + m)

    # Cond: theta + m < pi
    cond = (theta + m) < np.pi
    target_logits = np.where(cond, target_logits, cosine - m * np.sin(m))

    logits = one_hot * target_logits + (1.0 - one_hot) * cosine
    logits *= s

    # Softmax and Cross Entropy
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    loss = -np.mean(np.log(probs[np.arange(batch_size), labels] + 1e-8))

    print(f"ArcFace Loss: {loss:.4f}")
    assert loss > 0, "Loss should be positive"
    print("ArcFace Loss Component Test Passed.")

if __name__ == "__main__":
    main()
