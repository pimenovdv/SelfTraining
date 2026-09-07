import numpy as np
import argparse

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Training loop
def train_adam_ffn(X, y, hidden_size, epochs, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    input_size = X.shape[1]
    output_size = y.shape[1]

    # Initialize weights and biases randomly with mean 0
    np.random.seed(42) # For reproducibility
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))

    # Adam cache initialization
    m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)

    for epoch in range(1, epochs + 1):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Loss calculation (Mean Squared Error)
        loss = np.mean(0.5 * (a2 - y) ** 2)

        if epoch % (epochs // 10) == 0 or epoch == epochs:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

        # Backward pass
        # Error at output
        dZ2 = (a2 - y) * sigmoid_derivative(z2)
        dW2 = np.dot(a1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

        # Error at hidden layer
        dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(z1)
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

        # Adam updates
        t = epoch

        # Update W1
        m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
        v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1 ** 2)
        m_W1_hat = m_W1 / (1 - beta1 ** t)
        v_W1_hat = v_W1 / (1 - beta2 ** t)
        W1 -= learning_rate * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)

        # Update b1
        m_b1 = beta1 * m_b1 + (1 - beta1) * db1
        v_b1 = beta2 * v_b1 + (1 - beta2) * (db1 ** 2)
        m_b1_hat = m_b1 / (1 - beta1 ** t)
        v_b1_hat = v_b1 / (1 - beta2 ** t)
        b1 -= learning_rate * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)

        # Update W2
        m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
        v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2 ** 2)
        m_W2_hat = m_W2 / (1 - beta1 ** t)
        v_W2_hat = v_W2 / (1 - beta2 ** t)
        W2 -= learning_rate * m_W2_hat / (np.sqrt(v_W2_hat) + epsilon)

        # Update b2
        m_b2 = beta1 * m_b2 + (1 - beta1) * db2
        v_b2 = beta2 * v_b2 + (1 - beta2) * (db2 ** 2)
        m_b2_hat = m_b2 / (1 - beta1 ** t)
        v_b2_hat = v_b2 / (1 - beta2 ** t)
        b2 -= learning_rate * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)

    return W1, b1, W2, b2, a2

def main():
    parser = argparse.ArgumentParser(description="Train a simple FFN using Adam Optimizer.")
    parser.add_argument("--hidden_size", type=int, default=8, help="Number of neurons in hidden layer.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    args = parser.parse_args()

    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print(f"Training FFN with Adam (hidden_size={args.hidden_size}, epochs={args.epochs}, lr={args.lr})")

    W1, b1, W2, b2, predictions = train_adam_ffn(X, y, args.hidden_size, args.epochs, args.lr)

    print("\nTraining Complete.")
    print("Final Predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")

if __name__ == "__main__":
    main()
