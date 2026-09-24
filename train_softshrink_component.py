import numpy as np

class Softshrink:
    def __init__(self, lambd=0.5):
        self.lambd = lambd
        self.x = None

    def forward(self, x):
        self.x = x
        out = np.where(x > self.lambd, x - self.lambd, np.where(x < -self.lambd, x + self.lambd, 0.0))
        return out

    def backward(self, grad_output):
        grad_input = grad_output * np.where((self.x > self.lambd) | (self.x < -self.lambd), 1.0, 0.0)
        return grad_input

def test_component():
    np.random.seed(42)
    x = np.random.randn(10, 5)
    layer = Softshrink(lambd=0.5)

    # Forward pass
    out = layer.forward(x)
    print("Forward pass output shape:", out.shape)
    print("Forward pass output (first row):", out[0])

    # Backward pass
    grad_output = np.ones_like(out)
    grad_input = layer.backward(grad_output)
    print("Backward pass output shape:", grad_input.shape)
    print("Backward pass output (first row):", grad_input[0])

    print("Softshrink component testing completed successfully.")

if __name__ == "__main__":
    test_component()
