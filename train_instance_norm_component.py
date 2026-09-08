import numpy as np

def instance_norm(x, gamma, beta, eps=1e-5):
    # x shape: (N, C, H, W)
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2, 3), keepdims=True)
    x_normalized = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_normalized + beta
    return out

def main():
    print("Testing Instance Normalization...")
    np.random.seed(42)
    N, C, H, W = 2, 3, 4, 4
    x = np.random.randn(N, C, H, W)

    # Initialize gamma and beta
    gamma = np.ones((1, C, 1, 1))
    beta = np.zeros((1, C, 1, 1))

    out = instance_norm(x, gamma, beta)

    # Check if mean is close to 0 and variance is close to 1 for each instance and channel
    out_mean = np.mean(out, axis=(2, 3))
    out_var = np.var(out, axis=(2, 3))

    print(f"Output shape: {out.shape}")
    print(f"Mean across spatial dims:\n{out_mean}")
    print(f"Var across spatial dims:\n{out_var}")

    assert np.allclose(out_mean, 0, atol=1e-4)
    assert np.allclose(out_var, 1, atol=1e-4)

    print("Instance Normalization successful!")

if __name__ == "__main__":
    main()
