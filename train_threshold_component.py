import numpy as np

class Threshold:
    def __init__(self, threshold, value):
        self.threshold = threshold
        self.value = value

    def forward(self, x):
        self.x = x
        out = np.where(x > self.threshold, x, self.value)
        return out

    def backward(self, dout):
        dx = np.where(self.x > self.threshold, dout, 0.0)
        return dx

def test_component():
    np.random.seed(42)
    x = np.array([[-1.0, 0.5, 2.0], [3.0, -2.0, 1.0]])
    layer = Threshold(threshold=1.0, value=0.0)

    out = layer.forward(x)
    print("Forward output:")
    print(out)

    dout = np.ones_like(x)
    dx = layer.backward(dout)
    print("Backward output:")
    print(dx)

    print("Test passed.")

if __name__ == "__main__":
    test_component()
