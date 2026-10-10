import numpy as np

def aria2_forward(x, alpha=1.0, beta=1.0):
    """
    Aria-2 (Adaptive ReLU-inspired Activation 2) activation function.
    f(x) = x * (1 + alpha * x) / (1 + beta * x^2)
    This is an advanced mathematical activation function, useful for exploring
    smooth, non-monotonic properties.
    """
    return x * (1 + alpha * x) / (1 + beta * x**2)

def aria2_backward(x, alpha=1.0, beta=1.0):
    """
    Derivative of Aria-2 activation function.
    """
    denom = 1 + beta * x**2
    term1 = (1 + 2 * alpha * x) / denom
    term2 = (x * (1 + alpha * x) * 2 * beta * x) / (denom**2)
    return term1 - term2

def test_aria2():
    print("Testing Aria-2 Component...")
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Forward pass
    y = aria2_forward(x)
    print(f"Input: {x}")
    print(f"Output: {y}")

    # Backward pass
    grad = aria2_backward(x)
    print(f"Gradient: {grad}")

    # Basic assertions to ensure logic
    assert np.allclose(y[2], 0.0), "Aria-2 should be 0 at x=0"
    assert np.allclose(grad[2], 1.0), "Gradient should be 1 at x=0"
    print("Aria-2 Component mathematical tests passed!")

if __name__ == "__main__":
    test_aria2()
