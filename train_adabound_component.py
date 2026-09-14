import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class AdaBound:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, final_lr=0.1, gamma=1e-3):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.final_lr = final_lr
        self.gamma = gamma
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        # AdaBound specific bounds
        lower_bound = self.final_lr * (1 - 1 / (self.gamma * self.t + 1))
        upper_bound = self.final_lr * (1 + 1 / (self.gamma * self.t))

        for i in range(len(self.params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            step_size = lr_t / (np.sqrt(self.v[i]) + self.epsilon)

            # Apply dynamic bounds
            step_size = np.clip(step_size, lower_bound, upper_bound)

            self.params[i] -= step_size * self.m[i]

def train_adabound():
    # XOR dataset
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    # 2-layer network
    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))

    params = [W1, b1, W2, b2]
    optimizer = AdaBound(params, lr=0.01, final_lr=0.1)

    for epoch in range(5000):
        # Forward pass
        z1 = np.dot(X, params[0]) + params[1]
        a1 = sigmoid(z1)
        z2 = np.dot(a1, params[2]) + params[3]
        a2 = sigmoid(z2)

        # Loss (MSE)
        loss = np.mean(0.5 * (a2 - y)**2)

        # Backward pass
        d_a2 = (a2 - y)
        d_z2 = d_a2 * sigmoid_derivative(a2)
        d_W2 = np.dot(a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, params[2].T)
        d_z1 = d_a1 * sigmoid_derivative(a1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        grads = [d_W1, d_b1, d_W2, d_b2]

        optimizer.step(grads)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    print("Final Predictions:")
    print(np.round(a2, 2))
    assert loss < 0.1, "Model failed to converge"
    print("AdaBound Optimization successful.")

if __name__ == "__main__":
    train_adabound()
