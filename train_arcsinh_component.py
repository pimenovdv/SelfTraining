import numpy as np
import json
import os

class Arcsinh:
    def forward(self, x):
        self.x = x
        return np.arcsinh(x)

    def backward(self, dout):
        return dout / np.sqrt(self.x**2 + 1)

def test_component():
    np.random.seed(42)
    x = np.random.randn(10, 5)

    arcsinh = Arcsinh()
    out = arcsinh.forward(x)

    # Dummy gradient
    dout = np.ones_like(out)
    dx = arcsinh.backward(dout)

    # Simple training loop for regression
    w = np.random.randn(5, 1)
    b = np.random.randn(1)

    # Target
    y_true = np.random.randn(10, 1)

    learning_rate = 0.01
    losses = []

    for epoch in range(100):
        # Forward
        z = np.dot(x, w) + b
        a = arcsinh.forward(z)

        # MSE loss
        loss = np.mean((a - y_true)**2)
        losses.append(loss)

        # Backward
        da = 2 * (a - y_true) / y_true.size
        dz = arcsinh.backward(da)

        dw = np.dot(x.T, dz)
        db = np.sum(dz)

        # Update
        w -= learning_rate * dw
        b -= learning_rate * db

    results = {
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "status": "success" if losses[-1] < losses[0] else "failed"
    }

    os.makedirs("results", exist_ok=True)
    with open("results/arcsinh_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Arcsinh Component Test: Initial Loss = {results['initial_loss']:.4f}, Final Loss = {results['final_loss']:.4f}")
    return results

if __name__ == "__main__":
    test_component()
