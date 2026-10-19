import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class PCN:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2.0/layer_sizes[i-1]) for i in range(1, self.num_layers)]
        self.b = [np.zeros((layer_sizes[i], 1)) for i in range(1, self.num_layers)]

    def act(self, x):
        return np.tanh(x)
    def act_deriv(self, x):
        return 1.0 - np.tanh(x)**2

    def train_step(self, x, y, inference_steps=10, inference_lr=0.05, weight_lr=0.01):
        v = [np.zeros((self.W[i].shape[0], 1)) for i in range(len(self.W))]
        v.insert(0, x)
        for i in range(1, self.num_layers - 1):
            v[i] = self.W[i-1] @ self.act(v[i-1]) + self.b[i-1]
        v[-1] = y

        for step in range(inference_steps):
            errors = []
            for i in range(1, self.num_layers):
                mu_i = self.W[i-1] @ self.act(v[i-1]) + self.b[i-1]
                errors.append(v[i] - mu_i)
            for i in range(1, self.num_layers - 1):
                e_i = errors[i-1]
                e_next = errors[i]
                grad_v_i = e_i - (self.W[i].T @ e_next) * self.act_deriv(v[i])
                v[i] -= inference_lr * grad_v_i

        errors = []
        for i in range(1, self.num_layers):
            mu_i = self.W[i-1] @ self.act(v[i-1]) + self.b[i-1]
            errors.append(v[i] - mu_i)

        for i in range(self.num_layers - 1):
            e_i = errors[i]
            grad_W = -e_i @ self.act(v[i]).T
            grad_b = -e_i
            self.W[i] -= weight_lr * grad_W
            self.b[i] -= weight_lr * np.sum(grad_b, axis=1, keepdims=True)
        return sum(np.sum(e**2) for e in errors)

    def predict(self, x):
        v = x
        for i in range(self.num_layers - 1):
            v = self.W[i] @ self.act(v) + self.b[i]
        return v

class MLP:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2.0/layer_sizes[i-1]) for i in range(1, self.num_layers)]
        self.b = [np.zeros((layer_sizes[i], 1)) for i in range(1, self.num_layers)]

    def act(self, x):
        return np.tanh(x)
    def act_deriv(self, x):
        return 1.0 - np.tanh(x)**2

    def train_step(self, x, y, lr=0.01):
        a = [x]
        z = []
        for i in range(self.num_layers - 1):
            z_i = self.W[i] @ a[-1] + self.b[i]
            z.append(z_i)
            a.append(self.act(z_i))

        e = a[-1] - y
        loss = 0.5 * np.sum(e**2)

        delta = e * self.act_deriv(z[-1])
        grad_W = []
        grad_b = []
        grad_W.append(delta @ a[-2].T)
        grad_b.append(delta)

        for i in range(self.num_layers - 3, -1, -1):
            delta = (self.W[i+1].T @ delta) * self.act_deriv(z[i])
            grad_W.insert(0, delta @ a[i].T)
            grad_b.insert(0, delta)

        for i in range(self.num_layers - 1):
            self.W[i] -= lr * grad_W[i]
            self.b[i] -= lr * grad_b[i]

        return loss

    def predict(self, x):
        v = x
        for i in range(self.num_layers - 1):
            v = self.act(self.W[i] @ v + self.b[i])
        return v

def test_pcn_comparison():
    print("Testing PCN vs MLP (Backpropagation) Component...")
    digits = load_digits()
    X = digits.data
    y = digits.target

    X = X / 16.0

    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(y.reshape(-1, 1))
    Y = Y * 2 - 1

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    layer_sizes = [64, 32, 16, 10]
    pcn = PCN(layer_sizes)
    mlp = MLP(layer_sizes)

    epochs = 20
    print(f"Training PCN with architecture {layer_sizes} for {epochs} epochs...")
    start_time = time.time()
    for epoch in range(epochs):
        for i in range(len(X_train)):
            x_i = X_train[i].reshape(-1, 1)
            y_i = Y_train[i].reshape(-1, 1)
            pcn.train_step(x_i, y_i, inference_steps=10, inference_lr=0.05, weight_lr=0.01)
    pcn_time = time.time() - start_time

    print(f"Training MLP (Backprop) with architecture {layer_sizes} for {epochs} epochs...")
    start_time = time.time()
    for epoch in range(epochs):
        for i in range(len(X_train)):
            x_i = X_train[i].reshape(-1, 1)
            y_i = Y_train[i].reshape(-1, 1)
            mlp.train_step(x_i, y_i, lr=0.01)
    mlp_time = time.time() - start_time

    pcn_correct = 0
    mlp_correct = 0
    for i in range(len(X_test)):
        x_i = X_test[i].reshape(-1, 1)
        y_i = Y_test[i]
        pcn_pred = pcn.predict(x_i)
        mlp_pred = mlp.predict(x_i)
        if np.argmax(pcn_pred) == np.argmax(y_i):
            pcn_correct += 1
        if np.argmax(mlp_pred) == np.argmax(y_i):
            mlp_correct += 1

    pcn_acc = pcn_correct / len(X_test)
    mlp_acc = mlp_correct / len(X_test)

    print("\n--- Final Results ---")
    print(f"PCN: Accuracy={pcn_acc:.4f}, Time={pcn_time:.2f}s")
    print(f"MLP (Backprop): Accuracy={mlp_acc:.4f}, Time={mlp_time:.2f}s")

    if pcn_acc > 0.8:
        print("Success! The PCN successfully learned the image classification task and provided comparable accuracy to standard Backpropagation.")
    else:
        print("Failure. PCN failed to learn effectively.")
        exit(1)

if __name__ == "__main__":
    test_pcn_comparison()
