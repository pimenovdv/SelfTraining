import numpy as np
import json
import os

def tanhexp(x):
    """
    TanhExp activation function: f(x) = x * tanh(exp(x))
    """
    return x * np.tanh(np.exp(x))

def tanhexp_derivative(x):
    """
    Derivative of TanhExp: f'(x) = tanh(exp(x)) + x * exp(x) * (1 - tanh^2(exp(x)))
    """
    exp_x = np.exp(x)
    tanh_exp_x = np.tanh(exp_x)
    return tanh_exp_x + x * exp_x * (1.0 - tanh_exp_x**2)

def test_component():
    np.random.seed(42)
    x = np.random.randn(10, 5)

    # Forward pass
    output = tanhexp(x)

    # Backward pass (gradient)
    grad = tanhexp_derivative(x)

    results = {
        "x_mean": float(np.mean(x)),
        "output_mean": float(np.mean(output)),
        "grad_mean": float(np.mean(grad))
    }

    print(json.dumps(results, indent=2))

    os.makedirs("results", exist_ok=True)
    with open("results/tanhexp_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    test_component()
