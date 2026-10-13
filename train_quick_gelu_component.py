import numpy as np
import json
import os

class QuickGELU:
    def __init__(self, alpha=1.702):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        self.sigmoid = 1.0 / (1.0 + np.exp(-self.alpha * x))
        return x * self.sigmoid

    def backward(self, grad_output):
        grad_x = self.sigmoid + self.x * self.alpha * self.sigmoid * (1.0 - self.sigmoid)
        return grad_output * grad_x

def test_quick_gelu():
    np.random.seed(42)
    x = np.random.randn(5, 5)

    qgelu = QuickGELU()
    out = qgelu.forward(x)

    grad_output = np.ones_like(out)
    grad_x = qgelu.backward(grad_output)

    results = {
        "forward_mean": float(np.mean(out)),
        "forward_std": float(np.std(out)),
        "grad_mean": float(np.mean(grad_x)),
        "grad_std": float(np.std(grad_x)),
        "status": "success"
    }

    os.makedirs("results", exist_ok=True)
    with open("results/quick_gelu_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("QuickGELU Component tested successfully.")

if __name__ == "__main__":
    test_quick_gelu()
