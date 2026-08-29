import numpy as np

def sigmoid(x):
    # Clip to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class SqueezeAndExcitation:
    def __init__(self, channels, reduction=16):
        self.channels = channels
        self.reduced_channels = max(1, channels // reduction)

        # Initialize weights with He initialization for ReLU
        self.w1 = np.random.randn(self.reduced_channels, channels) * np.sqrt(2. / channels)
        self.b1 = np.zeros(self.reduced_channels)

        # Initialize weights with Xavier for Sigmoid
        self.w2 = np.random.randn(channels, self.reduced_channels) * np.sqrt(1. / self.reduced_channels)
        self.b2 = np.zeros(channels)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        self.x = x
        batch_size, c, h, w = x.shape

        # Squeeze: Global Average Pooling -> (batch_size, channels)
        self.z = np.mean(x, axis=(2, 3))

        # Excite
        self.fc1 = np.dot(self.z, self.w1.T) + self.b1
        self.a1 = relu(self.fc1)
        self.fc2 = np.dot(self.a1, self.w2.T) + self.b2
        self.s = sigmoid(self.fc2)

        # Scale: Broadcast s to match x
        self.s_reshaped = self.s[:, :, np.newaxis, np.newaxis]
        out = x * self.s_reshaped
        return out

    def backward(self, d_out, lr=0.01):
        batch_size, c, h, w = d_out.shape

        # d_out = dL/d(out)
        # out = x * s_reshaped
        d_x_from_scale = d_out * self.s_reshaped
        d_s_reshaped = d_out * self.x

        # Squeeze the gradients back to (batch_size, channels)
        d_s = np.sum(d_s_reshaped, axis=(2, 3))

        # Backprop through sigmoid
        d_fc2 = d_s * self.s * (1 - self.s)

        # Backprop through fc2
        d_w2 = np.dot(d_fc2.T, self.a1)
        d_b2 = np.sum(d_fc2, axis=0)
        d_a1 = np.dot(d_fc2, self.w2)

        # Backprop through relu
        d_fc1 = d_a1 * (self.fc1 > 0)

        # Backprop through fc1
        d_w1 = np.dot(d_fc1.T, self.z)
        d_b1 = np.sum(d_fc1, axis=0)
        d_z = np.dot(d_fc1, self.w1)

        # Backprop through Global Average Pooling
        # The gradient of mean is 1 / (H * W) spread across all elements
        d_x_from_squeeze = d_z[:, :, np.newaxis, np.newaxis] / (h * w)
        d_x_from_squeeze = np.broadcast_to(d_x_from_squeeze, self.x.shape)

        # Total gradient w.r.t input
        d_x = d_x_from_scale + d_x_from_squeeze

        # Update weights
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2
        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1

        return d_x

def test_se_block():
    print("Testing Squeeze-and-Excitation (SE) Block Component...")
    np.random.seed(42)

    # Mock data: (batch_size, channels, height, width)
    batch_size = 32
    channels = 16
    height = 8
    width = 8
    x = np.random.randn(batch_size, channels, height, width)

    # We want to train the SE block to output a specific channel scaling.
    # Note: SE block output scale `s` is between 0 and 1 due to sigmoid.
    # So our target scaling MUST be between 0 and 1.
    target_scale = np.ones((batch_size, channels))
    target_scale[:, :8] = 0.9
    target_scale[:, 8:] = 0.1
    y = x * target_scale[:, :, np.newaxis, np.newaxis]

    se_block = SqueezeAndExcitation(channels, reduction=4)

    epochs = 1000
    lr = 0.1

    for epoch in range(epochs):
        # Forward pass
        out = se_block.forward(x)

        # Loss (MSE)
        loss = np.mean((out - y)**2)

        # Backward pass
        d_out = 2 * (out - y) / (batch_size * channels * height * width)
        se_block.backward(d_out, lr=lr)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    # Final check
    final_out = se_block.forward(x)
    final_loss = np.mean((final_out - y)**2)
    print(f"Final Loss: {final_loss:.6f}")
    assert final_loss < 0.05, "SE block failed to learn the channel scaling."
    print("SE Block Component Test Passed.")

if __name__ == "__main__":
    test_se_block()
