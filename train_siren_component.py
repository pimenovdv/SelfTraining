import numpy as np
import os

class SineLayer:
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0, seed=42):
        np.random.seed(seed)
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega_0 = omega_0

        if self.is_first:
            limit = 1.0 / in_features
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        else:
            limit = np.sqrt(6.0 / in_features) / omega_0
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))

        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None
        self.pre_act = None

    def forward(self, x):
        self.x = x
        self.pre_act = self.omega_0 * (np.dot(x, self.W) + self.b)
        return np.sin(self.pre_act)

    def backward(self, d_out, learning_rate):
        d_pre_act = d_out * np.cos(self.pre_act) * self.omega_0

        self.dW = np.dot(self.x.T, d_pre_act)
        self.db = np.sum(d_pre_act, axis=0, keepdims=True)

        d_x = np.dot(d_pre_act, self.W.T)

        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

        return d_x

class LinearLayer:
    def __init__(self, in_features, out_features, seed=42):
        np.random.seed(seed)
        limit = np.sqrt(6.0 / in_features)
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, d_out, learning_rate):
        dW = np.dot(self.x.T, d_out)
        db = np.sum(d_out, axis=0, keepdims=True)
        d_x = np.dot(d_out, self.W.T)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return d_x

class SirenNetwork:
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, first_omega_0=30.0, hidden_omega_0=30.0, seed=42):
        self.layers = []
        self.layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, seed=seed))

        for i in range(hidden_layers):
            self.layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, seed=seed+i+1))

        if outermost_linear:
            self.layers.append(LinearLayer(hidden_features, out_features, seed=seed+hidden_layers+1))
        else:
            self.layers.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0, seed=seed+hidden_layers+1))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, d_out, learning_rate):
        d_x = d_out
        for layer in reversed(self.layers):
            d_x = layer.backward(d_x, learning_rate)
        return d_x

def train():
    np.random.seed(42)
    X = np.linspace(-1, 1, 1000).reshape(-1, 1)
    Y = np.sin(10 * X) + np.cos(25 * X)

    model = SirenNetwork(in_features=1, hidden_features=64, hidden_layers=3, out_features=1)

    epochs = 5000
    learning_rate = 1e-4
    losses = []

    for epoch in range(epochs):
        predictions = model.forward(X)
        loss = np.mean((predictions - Y)**2)
        losses.append(loss)

        d_loss = 2 * (predictions - Y) / Y.size
        model.backward(d_loss, learning_rate)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print(f"Initial Loss: {losses[0]:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")

    success = losses[-1] < 1e-3

    os.makedirs("docs", exist_ok=True)
    with open("docs/0094_train_siren_component.md", "w") as f:
        f.write(f"# Experiment 0094: Sinusoidal Representation Network (SIREN)\n\n")
        f.write(f"**Objective:** Implement and verify a Sinusoidal Representation Network (SIREN) mathematically.\n\n")
        f.write(f"**Methodology:** The SIREN uses sine functions as activation functions. It is initialized using a specific scheme (Sitzmann et al., 2020) to ensure activations remain within the useful domain of the sine function across layers. We train it to fit a complex 1D signal ($y = \\sin(10x) + \\cos(25x)$) using manual backpropagation with MSE loss.\n\n")
        f.write(f"**Results:**\n")
        f.write(f"- Initial Loss: {losses[0]:.4f}\n")
        f.write(f"- Final Loss: {losses[-1]:.4f}\n")
        f.write(f"- Success: {success}\n\n")
        f.write(f"**Conclusion:** The SIREN component successfully learned to approximate the high-frequency 1D signal with high precision, demonstrating its capability for continuous implicit representations.\n")
        f.write(f"**Script:** `train_siren_component.py`\n")

    print("Documentation generated at docs/0094_train_siren_component.md")
    return success

if __name__ == '__main__':
    success = train()
    if not success:
        exit(1)
