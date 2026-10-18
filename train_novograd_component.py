"""
Component Testing: NovoGrad Optimizer
This script implements the NovoGrad optimizer and trains a simple neural network
on the XOR dataset to verify its effectiveness.
"""

import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Test NovoGrad Component")
    parser.add_argument('--hidden_size', type=int, default=4, help='Hidden layer size')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.95, help='Beta1 parameter for NovoGrad')
    parser.add_argument('--beta2', type=float, default=0.98, help='Beta2 parameter for NovoGrad')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon parameter for NovoGrad')
    args = parser.parse_args()

    # Simple non-linear dataset (XOR)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    W1 = np.random.randn(2, args.hidden_size) * 0.1
    b1 = np.zeros((1, args.hidden_size))
    W2 = np.random.randn(args.hidden_size, 1) * 0.1
    b2 = np.zeros((1, 1))

    # NovoGrad initialization
    m_W1, v_W1 = np.zeros_like(W1), np.zeros((1,))
    m_b1, v_b1 = np.zeros_like(b1), np.zeros((1,))
    m_W2, v_W2 = np.zeros_like(W2), np.zeros((1,))
    m_b2, v_b2 = np.zeros_like(b2), np.zeros((1,))

    for epoch in range(args.epochs):
        # Forward pass
        Z1 = np.dot(X, W1) + b1
        A1 = np.maximum(0, Z1) # ReLU
        Z2 = np.dot(A1, W2) + b2
        A2 = 1 / (1 + np.exp(-Z2)) # Sigmoid

        loss = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))

        # Backward pass
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2) / len(X)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / len(X)
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(X.T, dZ1) / len(X)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / len(X)

        # NovoGrad updates
        # Layer 1 W
        grad_norm_W1 = np.linalg.norm(dW1)
        v_W1 = args.beta2 * v_W1 + (1 - args.beta2) * (grad_norm_W1 ** 2) if epoch > 0 else np.array([grad_norm_W1 ** 2])
        m_W1 = args.beta1 * m_W1 + (1 - args.beta1) * (dW1 / (np.sqrt(v_W1) + args.epsilon))
        W1 -= args.lr * m_W1

        # Layer 1 b
        grad_norm_b1 = np.linalg.norm(db1)
        v_b1 = args.beta2 * v_b1 + (1 - args.beta2) * (grad_norm_b1 ** 2) if epoch > 0 else np.array([grad_norm_b1 ** 2])
        m_b1 = args.beta1 * m_b1 + (1 - args.beta1) * (db1 / (np.sqrt(v_b1) + args.epsilon))
        b1 -= args.lr * m_b1

        # Layer 2 W
        grad_norm_W2 = np.linalg.norm(dW2)
        v_W2 = args.beta2 * v_W2 + (1 - args.beta2) * (grad_norm_W2 ** 2) if epoch > 0 else np.array([grad_norm_W2 ** 2])
        m_W2 = args.beta1 * m_W2 + (1 - args.beta1) * (dW2 / (np.sqrt(v_W2) + args.epsilon))
        W2 -= args.lr * m_W2

        # Layer 2 b
        grad_norm_b2 = np.linalg.norm(db2)
        v_b2 = args.beta2 * v_b2 + (1 - args.beta2) * (grad_norm_b2 ** 2) if epoch > 0 else np.array([grad_norm_b2 ** 2])
        m_b2 = args.beta1 * m_b2 + (1 - args.beta1) * (db2 / (np.sqrt(v_b2) + args.epsilon))
        b2 -= args.lr * m_b2

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Final Loss:", loss)
    assert loss < 0.1, f"Expected loss < 0.1, got {loss}"
    print("NovoGrad Component Test Passed.")

if __name__ == "__main__":
    main()
