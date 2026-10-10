import numpy as np

def gaussian_activation(x):
    return np.exp(-x**2)

def gaussian_activation_derivative(x):
    return -2 * x * np.exp(-x**2)

def test_gaussian_activation():
    x = np.array([-1.0, 0.0, 1.0])
    y = gaussian_activation(x)
    dy = gaussian_activation_derivative(x)

    assert np.allclose(y, np.array([np.exp(-1), 1.0, np.exp(-1)]))
    assert np.allclose(dy, np.array([2 * np.exp(-1), 0.0, -2 * np.exp(-1)]))

if __name__ == '__main__':
    test_gaussian_activation()
    print("Gaussian Activation Component tests passed!")
