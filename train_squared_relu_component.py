import numpy as np

def squared_relu(x):
    """
    Squared ReLU activation function.
    f(x) = max(0, x)^2
    """
    return np.maximum(0, x)**2

def squared_relu_derivative(x):
    """
    Derivative of Squared ReLU.
    f'(x) = 2 * max(0, x)
    """
    return 2 * np.maximum(0, x)

def test_squared_relu():
    np.random.seed(42)
    x = np.linspace(-3, 3, 7)

    output = squared_relu(x)
    grad = squared_relu_derivative(x)

    print("Inputs:", x)
    print("Squared ReLU Outputs:", output)
    print("Squared ReLU Gradients:", grad)

if __name__ == '__main__':
    test_squared_relu()
