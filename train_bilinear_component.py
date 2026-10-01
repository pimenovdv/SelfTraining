import numpy as np

def bilinear_layer(x1, x2, weight, bias=None):
    """
    Applies a bilinear transformation to the incoming data:
    y = x1^T W x2 + b

    Args:
        x1: Input tensor of shape (N, in1_features)
        x2: Input tensor of shape (N, in2_features)
        weight: Weight tensor of shape (out_features, in1_features, in2_features)
        bias: Optional bias tensor of shape (out_features,)

    Returns:
        Output tensor of shape (N, out_features)
    """
    out_features = weight.shape[0]
    batch_size = x1.shape[0]

    out = np.zeros((batch_size, out_features))

    for k in range(out_features):
        out[:, k] = np.sum(np.dot(x1, weight[k]) * x2, axis=1)

    if bias is not None:
        out += bias

    return out

def test_bilinear_component():
    np.random.seed(42)

    # Define dimensions
    batch_size = 2
    in1_features = 3
    in2_features = 4
    out_features = 5

    # Generate random inputs and weights
    x1 = np.random.randn(batch_size, in1_features)
    x2 = np.random.randn(batch_size, in2_features)
    weight = np.random.randn(out_features, in1_features, in2_features)
    bias = np.random.randn(out_features)

    print(f"x1 shape: {x1.shape}")
    print(f"x2 shape: {x2.shape}")
    print(f"weight shape: {weight.shape}")
    print(f"bias shape: {bias.shape}")

    # Forward pass
    output = bilinear_layer(x1, x2, weight, bias)

    print(f"Output shape: {output.shape}")
    print("Output values:")
    print(output)

    # Check expected shape
    assert output.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {output.shape}"

    print("\nBilinear Layer component successfully implemented and tested.")

if __name__ == "__main__":
    test_bilinear_component()
