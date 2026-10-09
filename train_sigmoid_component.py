import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of Sigmoid.
    f'(x) = sigmoid(x) * (1 - sigmoid(x))
    """
    s = sigmoid(x)
    return s * (1.0 - s)

def test_component():
    np.random.seed(42)
    x = np.linspace(-4, 4, 9)

    output = sigmoid(x)
    grad = sigmoid_derivative(x)

    print("Inputs:", x)
    print("Sigmoid Outputs:", output)
    print("Sigmoid Gradients:", grad)

if __name__ == '__main__':
    test_component()
