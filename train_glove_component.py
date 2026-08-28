import numpy as np

def train_glove():
    print("Training GloVe Component...")
    np.random.seed(42)
    vocab_size = 10
    embedding_dim = 4

    # Fake co-occurrence matrix X
    X = np.random.randint(0, 100, (vocab_size, vocab_size))
    np.fill_diagonal(X, 0)

    # Initialize weights
    W = np.random.randn(vocab_size, embedding_dim) * 0.1
    W_tilde = np.random.randn(vocab_size, embedding_dim) * 0.1
    b = np.random.randn(vocab_size, 1) * 0.1
    b_tilde = np.random.randn(vocab_size, 1) * 0.1

    # Hyperparameters
    x_max = 100
    alpha = 0.75
    lr = 0.005
    epochs = 500

    def f_weight(x):
        return (x / x_max)**alpha if x < x_max else 1.0

    for epoch in range(epochs):
        loss = 0
        for i in range(vocab_size):
            for j in range(vocab_size):
                if X[i, j] == 0:
                    continue

                weight = f_weight(X[i, j])
                diff = np.dot(W[i], W_tilde[j]) + b[i, 0] + b_tilde[j, 0] - np.log(X[i, j])

                loss += weight * (diff ** 2)

                # Gradients
                grad_diff = 2 * weight * diff

                grad_W_i = grad_diff * W_tilde[j]
                grad_W_tilde_j = grad_diff * W[i]
                grad_b_i = grad_diff
                grad_b_tilde_j = grad_diff

                # Updates
                W[i] -= lr * grad_W_i
                W_tilde[j] -= lr * grad_W_tilde_j
                b[i, 0] -= lr * grad_b_i
                b_tilde[j, 0] -= lr * grad_b_tilde_j

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print(f"Final Loss: {loss:.4f}")
    print("GloVe Component Training Completed.")
    return W

if __name__ == "__main__":
    train_glove()
