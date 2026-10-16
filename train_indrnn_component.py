import numpy as np

def train():
    np.random.seed(42)

    # Independent Recurrent Neural Network (IndRNN)
    # Allows deep RNNs without exploding/vanishing gradients by keeping
    # neurons independent in the same layer.

    input_size = 2
    hidden_size = 4
    seq_len = 5
    batch_size = 3
    epochs = 100

    X = np.random.randn(batch_size, seq_len, input_size)
    y = np.random.randn(batch_size, hidden_size) # Target for final state

    W_ih = np.random.randn(input_size, hidden_size) * 0.1
    # U is a vector, representing diagonal weight matrix for recurrent connections
    u_hh = np.random.uniform(-1, 1, hidden_size)
    b = np.zeros(hidden_size)

    lr = 0.01

    for epoch in range(epochs):
        h = np.zeros((batch_size, seq_len + 1, hidden_size))

        # Forward pass
        for t in range(seq_len):
            x_t = X[:, t, :]
            # Element-wise multiplication for recurrent part: u_hh * h[:, t, :]
            z_t = x_t @ W_ih + h[:, t, :] * u_hh + b
            h[:, t + 1, :] = np.tanh(z_t)

        final_h = h[:, -1, :]
        loss = np.mean((final_h - y) ** 2)

        # Backward pass
        dh_out = 2 * (final_h - y) / batch_size

        dW_ih = np.zeros_like(W_ih)
        du_hh = np.zeros_like(u_hh)
        db = np.zeros_like(b)

        dh_next = dh_out

        for t in reversed(range(seq_len)):
            x_t = X[:, t, :]
            h_prev = h[:, t, :]
            h_t = h[:, t + 1, :]

            dtanh = dh_next * (1 - h_t ** 2)

            dW_ih += x_t.T @ dtanh
            # Gradient for U is element-wise multiplication sum over batch
            du_hh += np.sum(h_prev * dtanh, axis=0)
            db += np.sum(dtanh, axis=0)

            # Gradient wrt previous hidden state (only through U)
            dh_next = dtanh * u_hh

        W_ih -= lr * dW_ih
        u_hh -= lr * du_hh
        b -= lr * db

    print(f"Final loss: {loss:.4f}")
    assert loss < 1.0, "IndRNN training failed"
    print("Success")

if __name__ == "__main__":
    train()
