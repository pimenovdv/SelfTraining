import numpy as np

class SpatialDropout2D:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        # x shape is (batch_size, channels, height, width)
        if training:
            # We want to drop entire channels
            # So mask shape is (batch_size, channels, 1, 1)
            mask_shape = (x.shape[0], x.shape[1], 1, 1)
            self.mask = np.random.binomial(1, 1.0 - self.p, size=mask_shape) / (1.0 - self.p)
            return x * self.mask
        return x

    def backward(self, grad_output):
        return grad_output * self.mask

def test_spatial_dropout():
    np.random.seed(42)
    x = np.ones((2, 3, 4, 4)) # 2 batches, 3 channels, 4x4 spatial

    dropout = SpatialDropout2D(p=0.5)

    # Forward pass
    out = dropout.forward(x, training=True)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Output (first batch, all channels, top-left pixel):")
    print(out[0, :, 0, 0])

    # Backward pass
    grad_output = np.ones_like(x)
    grad_x = dropout.backward(grad_output)
    print("Grad (first batch, all channels, top-left pixel):")
    print(grad_x[0, :, 0, 0])

    # Check that spatial elements in the same channel have the same mask
    for b in range(out.shape[0]):
        for c in range(out.shape[1]):
            # all elements in out[b, c, :, :] should be equal
            val = out[b, c, 0, 0]
            assert np.all(out[b, c, :, :] == val), "Spatial dropout failed to drop entire channels."

    print("Mathematical properties verified: Entire channels are correctly dropped or scaled.")

if __name__ == "__main__":
    test_spatial_dropout()
