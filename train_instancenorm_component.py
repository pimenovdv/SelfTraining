import numpy as np

def instance_norm(x, gamma, beta, eps=1e-5):
    # x shape: (N, C, H, W)
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2, 3), keepdims=True)

    x_normalized = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_normalized + beta

    cache = (x, x_normalized, mean, var, eps, gamma)
    return out, cache

def instance_norm_backward(dout, cache):
    x, x_normalized, mean, var, eps, gamma = cache
    N, C, H, W = x.shape

    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    dx_normalized = dout * gamma
    dvar = np.sum(dx_normalized * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=(2, 3), keepdims=True)
    dmean = np.sum(dx_normalized * -1.0 / np.sqrt(var + eps), axis=(2, 3), keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=(2, 3), keepdims=True)

    dx = dx_normalized / np.sqrt(var + eps) + dvar * 2.0 * (x - mean) / (H * W) + dmean / (H * W)

    return dx, dgamma, dbeta

def test_instancenorm_component():
    np.random.seed(42)
    N, C, H, W = 2, 3, 4, 4
    x = np.random.randn(N, C, H, W)
    gamma = np.ones((1, C, 1, 1))
    beta = np.zeros((1, C, 1, 1))

    out, cache = instance_norm(x, gamma, beta)

    out_mean = np.mean(out, axis=(2, 3))
    out_var = np.var(out, axis=(2, 3))

    assert np.allclose(out_mean, 0, atol=1e-5), f"Mean should be 0, got {out_mean}"
    assert np.allclose(out_var, 1, atol=1e-5), f"Variance should be 1, got {out_var}"

    dout = np.random.randn(N, C, H, W)
    dx, dgamma, dbeta = instance_norm_backward(dout, cache)

    assert dx.shape == x.shape
    assert dgamma.shape == gamma.shape
    assert dbeta.shape == beta.shape

    print("Instance Normalization forward and backward passed successfully!")

if __name__ == "__main__":
    test_instancenorm_component()
