import numpy as np

def relu6(x):
    """
    ReLU6 activation function.
    f(x) = min(max(0, x), 6)
    """
    return np.minimum(np.maximum(0, x), 6)

def relu6_derivative(x):
    """
    Derivative of ReLU6.
    f'(x) = 1 if 0 < x < 6 else 0
    """
    return np.where((x > 0) & (x < 6), 1.0, 0.0)

def test_relu6():
    np.random.seed(42)
    # Generate some random inputs from -2 to 8 to cover all regions
    x = np.linspace(-2, 8, 11)

    # Forward pass
    output = relu6(x)

    # Backward pass (derivative)
    grad = relu6_derivative(x)

    print("Inputs:", x)
    print("ReLU6 Outputs:", output)
    print("ReLU6 Gradients:", grad)

if __name__ == '__main__':
    test_relu6()
