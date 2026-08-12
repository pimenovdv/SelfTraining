import argparse
import numpy as np

def train_nmf(n_samples, n_features, n_components, epochs):
    # synthetic non-negative data
    np.random.seed(42)
    W_true = np.random.rand(n_samples, n_components)
    H_true = np.random.rand(n_components, n_features)
    V = W_true @ H_true

    # Initialize W and H randomly
    W = np.random.rand(n_samples, n_components)
    H = np.random.rand(n_components, n_features)

    for epoch in range(epochs):
        # Update H
        H = H * (W.T @ V) / (W.T @ W @ H + 1e-9)
        # Update W
        W = W * (V @ H.T) / (W @ H @ H.T + 1e-9)

        if epoch % max(1, epochs // 10) == 0:
            loss = np.linalg.norm(V - W @ H, 'fro')
            print(f"Epoch {epoch}: Reconstruction Loss = {loss:.4f}")

    final_loss = np.linalg.norm(V - W @ H, 'fro')
    print(f"Final Reconstruction Loss = {final_loss:.4f}")
    assert final_loss < 1e-1, f"NMF failed to converge adequately. Loss: {final_loss:.4f}"
    print("NMF component successfully implemented and verified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Non-negative Matrix Factorization (NMF) component.")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--n_features", type=int, default=10, help="Number of features")
    parser.add_argument("--n_components", type=int, default=3, help="Number of components (latent dimension)")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    args = parser.parse_args()

    train_nmf(args.n_samples, args.n_features, args.n_components, args.epochs)
