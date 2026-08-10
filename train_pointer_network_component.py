"""
Pointer Network
"""
import numpy as np
import os

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class PointerNetwork:
    """
    A Pointer Network that uses attention to select outputs from its inputs.
    """
    def __init__(self, input_size, hidden_size):
        self.H = hidden_size
        self.W_xh = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.b_h = np.zeros((1, hidden_size))

        self.W_xd = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.W_dd = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.b_d = np.zeros((1, hidden_size))

        self.W_1 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.W_2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.v = np.random.randn(hidden_size, 1) * np.sqrt(2.0/hidden_size)
        self.b_a = np.zeros((1, hidden_size))

        self.m = {k: np.zeros_like(getattr(self, k)) for k in ['W_xh', 'W_hh', 'b_h', 'W_xd', 'W_dd', 'b_d', 'W_1', 'W_2', 'v', 'b_a']}
        self.v_cache = {k: np.zeros_like(getattr(self, k)) for k in ['W_xh', 'W_hh', 'b_h', 'W_xd', 'W_dd', 'b_d', 'W_1', 'W_2', 'v', 'b_a']}
        self.t = 0

    def forward(self, X, Y_tf):
        B, T_in, _ = X.shape
        _, T_out, _ = Y_tf.shape
        self.B, self.T_in, self.T_out = B, T_in, T_out
        self.X, self.Y_tf = X, Y_tf

        self.enc_h = np.zeros((B, T_in, self.H))
        h_prev = np.zeros((B, self.H))
        for t in range(T_in):
            a = np.dot(X[:, t, :], self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
            self.enc_h[:, t, :] = np.tanh(a)
            h_prev = self.enc_h[:, t, :]

        self.dec_h = np.zeros((B, T_out, self.H))
        self.attn_scores = np.zeros((B, T_out, T_in))
        self.attn_probs = np.zeros((B, T_out, T_in))
        self.attn_hidden = np.zeros((B, T_out, T_in, self.H))

        d_prev = np.zeros((B, self.H))
        for t in range(T_out):
            a = np.dot(Y_tf[:, t, :], self.W_xd) + np.dot(d_prev, self.W_dd) + self.b_d
            self.dec_h[:, t, :] = np.tanh(a)
            d_prev = self.dec_h[:, t, :]

            for i in range(T_in):
                self.attn_hidden[:, t, i, :] = np.tanh(
                    np.dot(self.enc_h[:, i, :], self.W_1) +
                    np.dot(self.dec_h[:, t, :], self.W_2) + self.b_a
                )
                self.attn_scores[:, t, i] = np.dot(self.attn_hidden[:, t, i, :], self.v).squeeze(-1)

            self.attn_probs[:, t, :] = softmax(self.attn_scores[:, t, :])

        return self.attn_probs

    def backward(self, d_probs, lr=0.001):
        dW_xh, dW_hh, db_h = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.b_h)
        dW_xd, dW_dd, db_d = np.zeros_like(self.W_xd), np.zeros_like(self.W_dd), np.zeros_like(self.b_d)
        dW_1, dW_2, db_a = np.zeros_like(self.W_1), np.zeros_like(self.W_2), np.zeros_like(self.b_a)
        dv = np.zeros_like(self.v)

        d_scores = np.zeros_like(self.attn_scores)
        for t in range(self.T_out):
            for b in range(self.B):
                p = self.attn_probs[b, t, :].reshape(-1, 1)
                jac = np.diagflat(p) - np.dot(p, p.T)
                d_scores[b, t, :] = np.dot(jac, d_probs[b, t, :])

        d_dec_h = np.zeros_like(self.dec_h)
        d_enc_h = np.zeros_like(self.enc_h)

        for t in range(self.T_out):
            for i in range(self.T_in):
                ds = d_scores[:, t, i:i+1]
                dv += np.sum(self.attn_hidden[:, t, i, :].reshape(self.B, self.H, 1) * ds.reshape(self.B, 1, 1), axis=0)
                d_attn_hidden = np.dot(ds, self.v.T) * (1 - self.attn_hidden[:, t, i, :]**2)
                db_a += np.sum(d_attn_hidden, axis=0, keepdims=True)
                dW_1 += np.dot(self.enc_h[:, i, :].T, d_attn_hidden)
                dW_2 += np.dot(self.dec_h[:, t, :].T, d_attn_hidden)
                d_enc_h[:, i, :] += np.dot(d_attn_hidden, self.W_1.T)
                d_dec_h[:, t, :] += np.dot(d_attn_hidden, self.W_2.T)

        d_prev = np.zeros((self.B, self.H))
        for t in reversed(range(self.T_out)):
            dh = d_dec_h[:, t, :] + d_prev
            da = dh * (1 - self.dec_h[:, t, :]**2)
            dW_xd += np.dot(self.Y_tf[:, t, :].T, da)
            prev_dec_h = self.dec_h[:, t-1, :] if t > 0 else np.zeros((self.B, self.H))
            dW_dd += np.dot(prev_dec_h.T, da)
            db_d += np.sum(da, axis=0, keepdims=True)
            d_prev = np.dot(da, self.W_dd.T)

        h_prev_grad = np.zeros((self.B, self.H))
        for t in reversed(range(self.T_in)):
            dh = d_enc_h[:, t, :] + h_prev_grad
            da = dh * (1 - self.enc_h[:, t, :]**2)
            dW_xh += np.dot(self.X[:, t, :].T, da)
            prev_enc_h = self.enc_h[:, t-1, :] if t > 0 else np.zeros((self.B, self.H))
            dW_hh += np.dot(prev_enc_h.T, da)
            db_h += np.sum(da, axis=0, keepdims=True)
            h_prev_grad = np.dot(da, self.W_hh.T)

        grads = {'W_xh': dW_xh, 'W_hh': dW_hh, 'b_h': db_h, 'W_xd': dW_xd, 'W_dd': dW_dd, 'b_d': db_d, 'W_1': dW_1, 'W_2': dW_2, 'v': dv, 'b_a': db_a}

        # Adam Optimizer
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        for k in grads.keys():
            np.clip(grads[k], -5.0, 5.0, out=grads[k])
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * grads[k]
            self.v_cache[k] = beta2 * self.v_cache[k] + (1 - beta2) * (grads[k] ** 2)

            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v_cache[k] / (1 - beta2 ** self.t)

            param = getattr(self, k)
            param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

if __name__ == "__main__":
    np.random.seed(42)
    B = 32
    T = 5
    model = PointerNetwork(input_size=1, hidden_size=64)
    print("Training Pointer Network for Sorting Task...")

    final_loss = 0
    for epoch in range(1500):
        # Generate random sequences of floats
        X = np.random.rand(B, T, 1)
        targets = np.argsort(X[:, :, 0], axis=1) # (B, T)

        # Teacher Forcing Inputs
        Y_tf = np.zeros((B, T, 1))
        Y_tf[:, 0, 0] = -1.0 # start token
        for b in range(B):
            for t in range(1, T):
                Y_tf[b, t, 0] = X[b, targets[b, t-1], 0]

        probs = model.forward(X, Y_tf)

        loss = 0
        d_probs = np.zeros_like(probs)
        for b in range(B):
            for t in range(T):
                tgt = targets[b, t]
                p = probs[b, t, tgt]
                loss -= np.log(p + 1e-8)
                d_probs[b, t, tgt] = -1.0 / (p + 1e-8)

        loss /= (B * T)
        d_probs /= (B * T)
        final_loss = loss

        model.backward(d_probs, lr=0.005)

        if epoch % 200 == 0:
            preds = np.argmax(probs[0], axis=-1)
            print(f"Epoch {epoch} Loss: {loss:.4f} | Preds: {preds} Targets: {targets[0]}")

    print("Training complete.")
    success = final_loss < 0.5

    # Check what the next doc number is
    # We will let bash script figure out the next number
    # Generate the documentation
    doc_num = 128
    doc_path = f"docs/0128_train_pointer_network_component.md"

    with open(doc_path, "w") as f:
        f.write(f"# Experiment 0128: Pointer Network Component\n\n")
        f.write(f"**Objective:** Implement a Pointer Network in pure NumPy to learn conditional probabilities over an input dictionary, addressing tasks where the output vocabulary depends entirely on the input sequence (e.g., sorting).\n\n")
        f.write(f"**Script:** `train_pointer_network_component.py`\n\n")
        f.write(f"**Hypothesis:** By modifying the attention mechanism to output probabilities directly over the input sequence rather than blending encoder states, a neural network can successfully learn to point to input elements, enabling it to solve algorithmic tasks like sorting.\n\n")
        f.write(f"**Methodology:**\n")
        f.write(f"- Built an RNN encoder-decoder architecture.\n")
        f.write(f"- Implemented the pointer attention mechanism $u_i^t = v^T \\tanh(W_1 h_i + W_2 d_t)$.\n")
        f.write(f"- Applied softmax to $u^t$ to produce a probability distribution over the input sequence.\n")
        f.write(f"- Used Teacher Forcing during training on a sequence sorting task.\n")
        f.write(f"- Implemented manual backpropagation and the Adam optimizer.\n\n")
        f.write(f"**Results:**\n")
        if success:
            f.write(f"- **Status:** Success. The network converged to a low loss ({final_loss:.4f}), effectively learning to point to the correct sorted elements.\n")
        else:
            f.write(f"- **Status:** Failed. The network did not converge to a sufficiently low loss. Final loss: {final_loss:.4f}.\n")
        f.write(f"\n**Next Steps:**\n")
        f.write(f"- Evaluate the Pointer Network on combinatorial optimization problems like the Traveling Salesperson Problem (TSP).\n")

    print(f"Documentation generated at {doc_path}")
