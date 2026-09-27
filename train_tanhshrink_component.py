import numpy as np

def tanhshrink(x):
    """
    Applies the TanhShrink function element-wise.
    TanhShrink(x) = x - tanh(x)
    """
    return x - np.tanh(x)

def test_tanhshrink():
    print("Testing TanhShrink Component...")

    # Test values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Forward pass
    y = tanhshrink(x)

    print(f"Input x: {x}")
    print(f"Output y (TanhShrink(x)): {y}")

    # Check specific values
    assert np.allclose(y[2], 0.0), "TanhShrink(0) should be 0"

    print("Test passed successfully!")

if __name__ == "__main__":
    test_tanhshrink()
