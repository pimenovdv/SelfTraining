import numpy as np

def hardsigmoid(x):
    """
    Hardsigmoid activation function.
    f(x) = max(0, min(1, (x + 3) / 6))
    """
    return np.clip((x + 3.0) / 6.0, 0.0, 1.0)

def hardsigmoid_derivative(x):
    """
    Derivative of Hardsigmoid.
    f'(x) = 1/6 if -3 < x < 3 else 0
    """
    return np.where((x > -3.0) & (x < 3.0), 1.0 / 6.0, 0.0)

def test_component():
    np.random.seed(42)
    x = np.linspace(-4, 4, 9)

    output = hardsigmoid(x)
    grad = hardsigmoid_derivative(x)

    print("Inputs:", x)
    print("Hardsigmoid Outputs:", output)
    print("Hardsigmoid Gradients:", grad)

if __name__ == '__main__':
    test_component()
