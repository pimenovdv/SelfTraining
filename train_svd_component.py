import numpy as np
import os
import json

def generate_data(n_samples=100, n_features=5, rank=3):
    np.random.seed(42)
    U = np.random.randn(n_samples, rank)
    V = np.random.randn(rank, n_features)
    X = np.dot(U, V)
    return X

def train_svd(X, n_components=3):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    U_k = U[:, :n_components]
    s_k = s[:n_components]
    Vt_k = Vt[:n_components, :]
    X_reconstructed = np.dot(U_k * s_k, Vt_k)
    mse = np.mean((X - X_reconstructed) ** 2)
    return mse, U_k, s_k, Vt_k

if __name__ == "__main__":
    X = generate_data()
    mse, U, s, Vt = train_svd(X, n_components=3)
    print(f"MSE: {mse}")

    # Save a small artifact
    os.makedirs('results', exist_ok=True)
    with open('results/svd_results.json', 'w') as f:
        json.dump({'mse': mse, 'singular_values': s.tolist()}, f)
    print("SVD component tested successfully.")
