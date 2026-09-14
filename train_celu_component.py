import numpy as np

def celu(x, alpha=1.0):
    """
    Continuously Differentiable Exponential Linear Units (CELU)
    f(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
    """
    return np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))

def celu_derivative(x, alpha=1.0):
    """
    Derivative of CELU.
    f'(x) = 1 if x >= 0 else exp(x / alpha)
    """
    return np.where(x >= 0, 1.0, np.exp(x / alpha))

def test_celu():
    # Generate some random inputs
    np.random.seed(42)
    x = np.random.randn(10)

    # Forward pass
    output = celu(x)

    # Backward pass (derivative)
    grad = celu_derivative(x)

    print("Inputs:", x)
    print("CELU Outputs:", output)
    print("CELU Gradients:", grad)

if __name__ == '__main__':
    test_celu()
