import numpy as np
import os
import argparse

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class LiquidTimeConstantNetwork:
    """
    A minimal Liquid Time-Constant (LTC) Network layer.
    dx/dt = -(1/tau_sys + f(x, I)) * x + A * f(x, I)
    f is a sigmoid non-linearity depending on input and state.
    """
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W_rec = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b = np.zeros((1, hidden_dim))

        self.tau = np.random.uniform(0.5, 1.5, (1, hidden_dim))
        self.A = np.random.randn(1, hidden_dim) * 0.1

        self.W_out = np.random.randn(hidden_dim, 1) * 0.1
        self.b_out = np.zeros((1, 1))

    def forward(self, X_seq, dt=0.1):
        seq_len, batch_size, _ = X_seq.shape
        self.X_seq = X_seq
        self.dt = dt

        self.x_states = np.zeros((seq_len + 1, batch_size, self.hidden_dim))
        self.f_vals = np.zeros((seq_len, batch_size, self.hidden_dim))
        self.pre_f_vals = np.zeros((seq_len, batch_size, self.hidden_dim))

        for t in range(seq_len):
            pre_f = np.dot(X_seq[t], self.W) + np.dot(self.x_states[t], self.W_rec) + self.b
            self.pre_f_vals[t] = pre_f
            f = sigmoid(pre_f)
            self.f_vals[t] = f

            dx = -(1 / self.tau + f) * self.x_states[t] + self.A * f
            self.x_states[t+1] = self.x_states[t] + dt * dx

        self.out = np.dot(self.x_states[-1], self.W_out) + self.b_out
        return self.out

    def backward(self, d_out, lr=0.01):
        seq_len, batch_size, _ = self.X_seq.shape

        dW_out = np.dot(self.x_states[-1].T, d_out) / batch_size
        db_out = np.sum(d_out, axis=0, keepdims=True) / batch_size

        dx_state = np.dot(d_out, self.W_out.T)

        dW = np.zeros_like(self.W)
        dW_rec = np.zeros_like(self.W_rec)
        db = np.zeros_like(self.b)
        dtau = np.zeros_like(self.tau)
        dA = np.zeros_like(self.A)

        for t in reversed(range(seq_len)):
            # derivative from the Euler step
            dx_prev = dx_state * (1 - self.dt * (1 / self.tau + self.f_vals[t]))
            df = dx_state * self.dt * (-self.x_states[t] + self.A)

            dpre_f = df * sigmoid_derivative(self.pre_f_vals[t])

            dW += np.dot(self.X_seq[t].T, dpre_f) / batch_size
            dW_rec += np.dot(self.x_states[t].T, dpre_f) / batch_size
            db += np.sum(dpre_f, axis=0, keepdims=True) / batch_size

            # gradient from f dependency on x_t goes to dx_prev
            dx_prev += np.dot(dpre_f, self.W_rec.T)

            dtau += np.sum(dx_state * self.dt * (1 / (self.tau**2)) * self.x_states[t], axis=0, keepdims=True) / batch_size
            dA += np.sum(dx_state * self.dt * self.f_vals[t], axis=0, keepdims=True) / batch_size

            dx_state = dx_prev

        self.W -= lr * dW
        self.W_rec -= lr * dW_rec
        self.b -= lr * db
        self.tau -= lr * dtau
        self.A -= lr * dA
        self.W_out -= lr * dW_out
        self.b_out -= lr * db_out

def train_test():
    X = np.random.randn(200, 5, 1)
    Y = np.zeros((200, 1))
    for i in range(200):
        if np.sum(X[i, :, 0]) > 0:
            Y[i, 0] = 1.0
        else:
            Y[i, 0] = 0.0

    X_seq = np.transpose(X, (1, 0, 2))

    print("Training Liquid Time-Constant (LTC) Network...")
    model = LiquidTimeConstantNetwork(1, 16)

    for epoch in range(1000):
        out = model.forward(X_seq, dt=0.1)
        loss = np.mean((out - Y)**2)
        d_out = 2 * (out - Y)

        model.backward(d_out, lr=0.1)

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print(f"Final Loss: {loss:.6f}")

    if loss < 0.25:
        print("Success! Model learned temporal threshold via LTC.")
    else:
        print("Failed.")

if __name__ == "__main__":
    train_test()
