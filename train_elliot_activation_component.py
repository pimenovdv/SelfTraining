import numpy as np
import json
import os

def elliot_activation(x):
    """
    Elliot activation function: f(x) = 0.5 * x / (1 + |x|) + 0.5
    An approximation of the sigmoid function that avoids exponentials.
    """
    return 0.5 * x / (1 + np.abs(x)) + 0.5

def test_elliot_activation():
    print("Testing Elliot Activation Component...")

    # Test cases
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    expected = np.array([0.5 * -5.0 / (1 + 5.0) + 0.5,
                         0.5 * -1.0 / (1 + 1.0) + 0.5,
                         0.5 * 0.0 / (1 + 0.0) + 0.5,
                         0.5 * 1.0 / (1 + 1.0) + 0.5,
                         0.5 * 5.0 / (1 + 5.0) + 0.5])

    result = elliot_activation(x)

    print(f"Input: {x}")
    print(f"Expected: {expected}")
    print(f"Result: {result}")

    assert np.allclose(result, expected), "Elliot activation computation failed"
    print("Forward pass tests passed.")

    # Derivative check using finite differences
    eps = 1e-5
    x_test = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    # Analytical derivative: f'(x) = 0.5 / (1 + |x|)^2
    analytical_grad = 0.5 / (1 + np.abs(x_test))**2

    # Numerical derivative
    numerical_grad = (elliot_activation(x_test + eps) - elliot_activation(x_test - eps)) / (2 * eps)

    print(f"Analytical gradient: {analytical_grad}")
    print(f"Numerical gradient: {numerical_grad}")

    assert np.allclose(analytical_grad, numerical_grad, atol=1e-4), "Gradient computation failed"
    print("Gradient tests passed.")

    os.makedirs("results", exist_ok=True)
    with open("results/elliot_activation_results.json", "w") as f:
        json.dump({"status": "success", "message": "Elliot activation forward and backward passed"}, f)
    print("Component test completed successfully.")

if __name__ == "__main__":
    test_elliot_activation()
