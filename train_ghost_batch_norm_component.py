import numpy as np
import os
import json

def ghost_batch_norm(X, gamma, beta, num_splits=2, eps=1e-5):
    """
    Ghost Batch Normalization
    X: (batch_size, features)
    gamma, beta: (features,)
    num_splits: number of ghost batches
    """
    batch_size, features = X.shape
    assert batch_size % num_splits == 0, "Batch size must be divisible by num_splits"

    ghost_batch_size = batch_size // num_splits
    X_reshaped = X.reshape(num_splits, ghost_batch_size, features)

    mean = np.mean(X_reshaped, axis=1, keepdims=True)
    var = np.var(X_reshaped, axis=1, keepdims=True)

    X_norm = (X_reshaped - mean) / np.sqrt(var + eps)
    out = gamma * X_norm + beta

    return out.reshape(batch_size, features)

def test_component():
    np.random.seed(42)
    X = np.random.randn(16, 8)
    gamma = np.ones(8)
    beta = np.zeros(8)

    out = ghost_batch_norm(X, gamma, beta, num_splits=4)

    print("Ghost Batch Norm Output Shape:", out.shape)
    assert out.shape == (16, 8)
    print("Test passed!")

    os.makedirs("results", exist_ok=True)
    with open("results/ghost_batch_norm_results.json", "w") as f:
        json.dump({"shape": out.shape, "mean": float(np.mean(out))}, f)

if __name__ == "__main__":
    test_component()
