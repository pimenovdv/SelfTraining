import numpy as np

def silu(x):
    """Sigmoid Linear Unit (SiLU) activation function. Also known as Swish-1."""
    return x * (1 / (1 + np.exp(-x)))

def silu_derivative(x):
    """Derivative of SiLU with respect to x."""
    s = 1 / (1 + np.exp(-x))
    return s + x * s * (1 - s)

def train_component():
    """
    Test the SiLU component by passing it some values.
    """
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    print("Testing SiLU Component...")
    print(f"Input x:\n{x}")
    y = silu(x)
    print(f"SiLU(x):\n{y}")
    dy = silu_derivative(x)
    print(f"SiLU'(x):\n{dy}")
    print("SiLU component test completed successfully.")

    assert np.allclose(silu(np.array([0.0])), np.array([0.0])), "SiLU(0) should be 0"

if __name__ == "__main__":
    train_component()
