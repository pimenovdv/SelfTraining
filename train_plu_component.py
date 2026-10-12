import numpy as np

def plu(x, alpha=0.1, c=1.0):
    """
    Piecewise Linear Unit (PLU) activation function.
    f(x) = max(alpha * (x + c) - c, min(alpha * (x - c) + c, x))
    """
    return np.maximum(alpha * (x + c) - c, np.minimum(alpha * (x - c) + c, x))

def plu_derivative(x, alpha=0.1, c=1.0):
    """
    Derivative of Piecewise Linear Unit (PLU) activation function.
    """
    dx = np.ones_like(x)
    dx[x > c] = alpha
    dx[x < -c] = alpha
    return dx

def test_plu():
    print("Testing Piecewise Linear Unit (PLU) Component...")
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    alpha = 0.1
    c = 1.0

    y = plu(x, alpha=alpha, c=c)
    dy = plu_derivative(x, alpha=alpha, c=c)

    print(f"Input: {x}")
    print(f"Output: {y}")
    print(f"Derivative: {dy}")

    assert np.allclose(y, np.array([-1.1, -1.0, 0.0, 1.0, 1.1]))
    assert np.allclose(dy, np.array([0.1, 1.0, 1.0, 1.0, 0.1]))
    print("Test passed successfully.")

if __name__ == "__main__":
    test_plu()
