import numpy as np
import json
import os

def solu(x):
    """
    SoLU (Softmax Linear Unit) activation function.
    Proposed as an activation function to increase superposition and interpretability.
    Mathematically: SoLU(x) = x * softmax(x)
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return x * softmax_x

def test_component():
    np.random.seed(42)

    # Generate random input data
    x = np.random.randn(2, 5)

    # Apply SoLU activation
    output = solu(x)

    print("Input:\n", x)
    print("Output:\n", output)

    assert output.shape == x.shape, "Output shape should match input shape"

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/solu_results.json", "w") as f:
        json.dump({"input": x.tolist(), "output": output.tolist()}, f)

    print("Test passed successfully.")

if __name__ == "__main__":
    test_component()
