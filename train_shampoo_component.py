import numpy as np
import argparse

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Training loop
def train_shampoo_ffn(X, y, hidden_size, epochs, learning_rate, epsilon=1e-4):
    input_size = X.shape[1]
    output_size = y.shape[1]

    # Initialize weights and biases randomly with mean 0
    np.random.seed(42) # For reproducibility
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))

    # Shampoo preconditioners
    L_W1 = epsilon * np.eye(input_size)
    R_W1 = epsilon * np.eye(hidden_size)
    L_W2 = epsilon * np.eye(hidden_size)
    R_W2 = epsilon * np.eye(output_size)

    # Biases treated as 1D tensors, preconditioner is scalar
    L_b1 = epsilon
    L_b2 = epsilon

    def inverse_pth_root(A, p):
        # Compute A^{-1/p} for a positive definite matrix A
        # For small matrices, eigen decomposition is fine
        evals, evecs = np.linalg.eigh(A)
        # Ensure positive eigenvalues
        evals = np.maximum(evals, 1e-8)
        inv_roots = np.diag(evals ** (-1.0 / p))
        return evecs @ inv_roots @ evecs.T

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

        # Shampoo updates

        # W1 update
        L_W1 += np.dot(dW1, dW1.T)
        R_W1 += np.dot(dW1.T, dW1)
        L_W1_inv4 = inverse_pth_root(L_W1, 4)
        R_W1_inv4 = inverse_pth_root(R_W1, 4)
        preconditioned_dW1 = L_W1_inv4 @ dW1 @ R_W1_inv4
        W1 -= learning_rate * preconditioned_dW1

        # b1 update
        L_b1 += np.sum(db1**2)
        b1 -= learning_rate * db1 / (L_b1**(1/4))

        # W2 update
        L_W2 += np.dot(dW2, dW2.T)
        R_W2 += np.dot(dW2.T, dW2)
        L_W2_inv4 = inverse_pth_root(L_W2, 4)
        R_W2_inv4 = inverse_pth_root(R_W2, 4)
        preconditioned_dW2 = L_W2_inv4 @ dW2 @ R_W2_inv4
        W2 -= learning_rate * preconditioned_dW2

        # b2 update
        L_b2 += np.sum(db2**2)
        b2 -= learning_rate * db2 / (L_b2**(1/4))

    return W1, b1, W2, b2, a2

def main():
    parser = argparse.ArgumentParser(description="Train a simple FFN using Shampoo Optimizer.")
    parser.add_argument("--hidden_size", type=int, default=8, help="Number of neurons in hidden layer.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate.") # Shampoo often needs different lr
    args = parser.parse_args()

    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print(f"Training FFN with Shampoo (hidden_size={args.hidden_size}, epochs={args.epochs}, lr={args.lr})")

    W1, b1, W2, b2, predictions = train_shampoo_ffn(X, y, args.hidden_size, args.epochs, args.lr)

    print("\nTraining Complete.")
    print("Final Predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")

if __name__ == "__main__":
    main()
