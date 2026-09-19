import numpy as np

def glu(x, dim=-1):
    """
    Gated Linear Unit (GLU).
    Splits the input x into two halves along the given dimension,
    a and b, and returns a * sigmoid(b).
    """
    split_size = x.shape[dim] // 2
    a, b = np.split(x, [split_size], axis=dim)
    return a * (1 / (1 + np.exp(-b)))

def test_glu_component():
    np.random.seed(42)
    # Create random input tensor
    x = np.random.randn(2, 4)
    print("Input:\n", x)

    # Apply GLU
    out = glu(x, dim=-1)
    print("\nGLU Output:\n", out)

    assert out.shape == (2, 2), "Output shape should be half of the input along the split dimension"
    print("\nGLU component successfully implemented and tested.")

if __name__ == "__main__":
    test_glu_component()
