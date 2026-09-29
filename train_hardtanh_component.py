import numpy as np

def hardtanh(x, min_val=-1.0, max_val=1.0):
    """
    Hardtanh activation function.
    f(x) = max(min_val, min(max_val, x))
    """
    return np.clip(x, min_val, max_val)

def hardtanh_derivative(x, min_val=-1.0, max_val=1.0):
    """
    Derivative of Hardtanh.
    f'(x) = 1 if min_val < x < max_val else 0
    """
    return np.where((x > min_val) & (x < max_val), 1.0, 0.0)

def test_component():
    np.random.seed(42)
    x = np.linspace(-2, 2, 9)

    output = hardtanh(x)
    grad = hardtanh_derivative(x)

    print("Inputs:", x)
    print("Hardtanh Outputs:", output)
    print("Hardtanh Gradients:", grad)

if __name__ == '__main__':
    test_component()
