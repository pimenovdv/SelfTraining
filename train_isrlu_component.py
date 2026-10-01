import numpy as np

def isrlu(x, alpha=1.0):
    """
    Inverse Square Root Linear Unit (ISRLU)
    f(x) = x if x >= 0 else x / sqrt(1 + alpha * x^2)
    """
    return np.where(x >= 0, x, x / np.sqrt(1 + alpha * (x ** 2)))

def isrlu_derivative(x, alpha=1.0):
    """
    Derivative of ISRLU.
    f'(x) = 1 if x >= 0 else (1 / sqrt(1 + alpha * x^2))^3
    """
    return np.where(x >= 0, 1.0, (1 / np.sqrt(1 + alpha * (x ** 2))) ** 3)

def test_isrlu():
    # Generate some random inputs
    np.random.seed(42)
    x = np.random.randn(10)

    # Forward pass
    output = isrlu(x)

    # Backward pass (derivative)
    grad = isrlu_derivative(x)

    print("Inputs:", x)
    print("ISRLU Outputs:", output)
    print("ISRLU Gradients:", grad)

if __name__ == '__main__':
    test_isrlu()
