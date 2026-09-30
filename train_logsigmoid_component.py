import numpy as np
import json
import os

def logsigmoid(x):
    """
    LogSigmoid activation function.
    LogSigmoid(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
    """
    return np.where(x < 0, x - np.log1p(np.exp(x)), -np.log1p(np.exp(-x)))

def logsigmoid_derivative(x):
    """
    Derivative of LogSigmoid activation function.
    d/dx logsigmoid(x) = 1 / (1 + exp(x)) = 1 - sigmoid(x)
    """
    return 1.0 / (1.0 + np.exp(x))

def test_logsigmoid_component():
    print("Testing LogSigmoid Component...")

    # Test values
    x_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Expected values
    expected_y = np.where(x_val < 0, x_val - np.log1p(np.exp(x_val)), -np.log1p(np.exp(-x_val)))

    # Actual values
    y_val = logsigmoid(x_val)

    print(f"Input x: {x_val}")
    print(f"LogSigmoid(x): {y_val}")

    # Derivative
    dy_val = logsigmoid_derivative(x_val)
    print(f"LogSigmoid'(x): {dy_val}")

    # Check if implemented correctly
    assert np.allclose(y_val, expected_y), "LogSigmoid calculation is incorrect!"

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/logsigmoid_results.json", "w") as f:
        json.dump({
            "x": x_val.tolist(),
            "y": y_val.tolist(),
            "dy": dy_val.tolist()
        }, f, indent=4)

    print("LogSigmoid Component tests passed successfully.")

if __name__ == "__main__":
    test_logsigmoid_component()
