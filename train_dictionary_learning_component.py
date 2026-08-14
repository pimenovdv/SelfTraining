import numpy as np
import argparse

def soft_thresholding(v, lambda_):
    return np.sign(v) * np.maximum(np.abs(v) - lambda_, 0)

def fista(X, D, lambda_, max_iter=50):
    N, d = X.shape
    k = D.shape[0]

    alpha = np.zeros((N, k))
    y = np.zeros((N, k))
    t = 1.0

    # Lipschitz constant of the gradient
    L = np.linalg.norm(D @ D.T, 2)
    step_size = 1.0 / L if L > 0 else 1.0

    for _ in range(max_iter):
        alpha_prev = alpha.copy()

        # Gradient descent step
        grad = (y @ D - X) @ D.T
        alpha_unconstrained = y - step_size * grad

        # Proximal step (soft thresholding)
        alpha = soft_thresholding(alpha_unconstrained, step_size * lambda_)

        # Nesterov momentum update
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = alpha + ((t - 1) / t_next) * (alpha - alpha_prev)
        t = t_next

    return alpha

def train_dictionary_learning(args):
    np.random.seed(42)

    # Generate synthetic data
    N, d, k = args.num_samples, args.input_dim, args.num_atoms

    # True dictionary
    D_true = np.random.randn(k, d)
    D_true = D_true / np.linalg.norm(D_true, axis=1, keepdims=True)

    # True sparse codes
    alpha_true = np.zeros((N, k))
    for i in range(N):
        active = np.random.choice(k, max(1, int(0.1 * k)), replace=False)
        alpha_true[i, active] = np.random.randn(len(active))

    X = alpha_true @ D_true + 0.01 * np.random.randn(N, d)

    # Initialize dictionary randomly
    D = np.random.randn(k, d)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)

    print("Starting Dictionary Learning via Alternating Optimization...")
    for epoch in range(args.epochs):
        # 1. Sparse Coding step (Update alpha with D fixed)
        alpha = fista(X, D, args.lambda_, max_iter=20)

        # 2. Dictionary Update step (Update D with alpha fixed)
        grad_D = alpha.T @ (alpha @ D - X) / N
        D -= args.lr * grad_D

        # Project dictionary atoms to unit norm
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            recon = alpha @ D
            mse = np.mean((X - recon)**2)
            sparsity = np.mean(alpha == 0)
            print(f"Epoch {epoch:4d} | MSE: {mse:.6f} | Sparsity: {sparsity:.2%}")

    print("Dictionary Learning training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dictionary Learning Component")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--input_dim", type=int, default=20)
    parser.add_argument("--num_atoms", type=int, default=30)
    parser.add_argument("--lambda_", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0)
    args = parser.parse_args()

    train_dictionary_learning(args)
