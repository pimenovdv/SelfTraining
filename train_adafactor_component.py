import numpy as np
import argparse

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Training loop
def train_adafactor_ffn(X, y, hidden_size, epochs, learning_rate, beta2=0.999, epsilon=1e-30, d=1e-3):
    input_size = X.shape[1]
    output_size = y.shape[1]

    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))

    # Adafactor cache initialization
    # For matrices, we store row and column sums of exponentially decayed squared gradients
    v_r_W1 = np.zeros((input_size, 1))
    v_c_W1 = np.zeros((1, hidden_size))

    v_r_W2 = np.zeros((hidden_size, 1))
    v_c_W2 = np.zeros((1, output_size))

    # For vectors (biases), we just use standard RMSprop-like updates
    v_b1 = np.zeros_like(b1)
    v_b2 = np.zeros_like(b2)

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
        dZ2 = (a2 - y) * sigmoid_derivative(z2)
        dW2 = np.dot(a1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

        dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(z1)
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

        # Update W1 using Adafactor
        v_r_W1 = beta2 * v_r_W1 + (1 - beta2) * np.sum(dW1 ** 2, axis=1, keepdims=True)
        v_c_W1 = beta2 * v_c_W1 + (1 - beta2) * np.sum(dW1 ** 2, axis=0, keepdims=True)
        v_hat_W1 = (v_r_W1 @ v_c_W1) / (np.sum(v_r_W1) + epsilon)
        W1 -= learning_rate * dW1 / (np.sqrt(v_hat_W1) + d)

        # Update b1
        v_b1 = beta2 * v_b1 + (1 - beta2) * (db1 ** 2)
        b1 -= learning_rate * db1 / (np.sqrt(v_b1) + d)

        # Update W2 using Adafactor
        v_r_W2 = beta2 * v_r_W2 + (1 - beta2) * np.sum(dW2 ** 2, axis=1, keepdims=True)
        v_c_W2 = beta2 * v_c_W2 + (1 - beta2) * np.sum(dW2 ** 2, axis=0, keepdims=True)
        v_hat_W2 = (v_r_W2 @ v_c_W2) / (np.sum(v_r_W2) + epsilon)
        W2 -= learning_rate * dW2 / (np.sqrt(v_hat_W2) + d)

        # Update b2
        v_b2 = beta2 * v_b2 + (1 - beta2) * (db2 ** 2)
        b2 -= learning_rate * db2 / (np.sqrt(v_b2) + d)

    return W1, b1, W2, b2, a2

def main():
    parser = argparse.ArgumentParser(description="Train a simple FFN using Adafactor Optimizer.")
    parser.add_argument("--hidden_size", type=int, default=8, help="Number of neurons in hidden layer.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    args = parser.parse_args()

    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print(f"Training FFN with Adafactor (hidden_size={args.hidden_size}, epochs={args.epochs}, lr={args.lr})")

    W1, b1, W2, b2, predictions = train_adafactor_ffn(X, y, args.hidden_size, args.epochs, args.lr)

    print("\nTraining Complete.")
    print("Final Predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")

if __name__ == "__main__":
    main()
