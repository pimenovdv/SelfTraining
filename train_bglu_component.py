import numpy as np
import os
import json

class BilinearGatedLinearUnit:
    def __init__(self, in_features, out_features):
        self.W_linear = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b_linear = np.zeros(out_features)
        self.W_gate = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b_gate = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        self.linear_out = x @ self.W_linear + self.b_linear
        self.gate_out = x @ self.W_gate + self.b_gate
        return self.linear_out * self.gate_out

    def backward(self, dout, lr=0.01):
        x = self.x
        d_linear_out = dout * self.gate_out
        d_gate_out = dout * self.linear_out

        dW_linear = x.T @ d_linear_out
        db_linear = np.sum(d_linear_out, axis=0)

        dW_gate = x.T @ d_gate_out
        db_gate = np.sum(d_gate_out, axis=0)

        dx = d_linear_out @ self.W_linear.T + d_gate_out @ self.W_gate.T

        self.W_linear -= lr * dW_linear
        self.b_linear -= lr * db_linear
        self.W_gate -= lr * dW_gate
        self.b_gate -= lr * db_gate

        return dx

def test_bglu():
    print("Testing BGLU Component...")
    np.random.seed(42)

    batch_size = 32
    in_features = 64
    out_features = 32

    model = BilinearGatedLinearUnit(in_features, out_features)

    x = np.random.randn(batch_size, in_features)
    target = np.random.randn(batch_size, out_features)

    epochs = 100
    lr = 0.01

    losses = []

    for epoch in range(epochs):
        out = model.forward(x)
        loss = np.mean((out - target) ** 2)
        losses.append(loss)

        dout = 2 * (out - target) / batch_size
        model.backward(dout, lr=lr)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print(f"Final Loss: {losses[-1]:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/bglu_results.json", "w") as f:
        json.dump({"final_loss": float(losses[-1])}, f)

if __name__ == "__main__":
    test_bglu()
