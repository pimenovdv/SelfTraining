import numpy as np
import argparse
import os

def get_xpos_scale_base(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    dim = np.arange(0, d_model, 2)
    gamma = 0.9 + 0.1 * (dim / d_model)
    scales = gamma ** pos
    return scales

def get_rope_cos_sin(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    dim = np.arange(0, d_model, 2)
    freqs = 1.0 / (10000 ** (dim / d_model))
    theta = pos * freqs
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta, sin_theta

def apply_xpos(x, cos_theta, sin_theta, scales, is_query=True):
    x1 = x[:, 0::2]
    x2 = x[:, 1::2]
    out = np.empty_like(x)

    # Paper notes: Q * gamma**n, K * gamma**-m
    if is_query:
        x1 = x1 * scales
        x2 = x2 * scales
    else:
        x1 = x1 / scales
        x2 = x2 / scales

    out[:, 0::2] = x1 * cos_theta - x2 * sin_theta
    out[:, 1::2] = x2 * cos_theta + x1 * sin_theta

    return out

def apply_xpos_backward(dOut, cos_theta, sin_theta, scales, is_query=True):
    dOut1 = dOut[:, 0::2]
    dOut2 = dOut[:, 1::2]

    dX1 = dOut1 * cos_theta + dOut2 * sin_theta
    dX2 = dOut2 * cos_theta - dOut1 * sin_theta

    if is_query:
        dX1 = dX1 * scales
        dX2 = dX2 * scales
    else:
        dX1 = dX1 / scales
        dX2 = dX2 / scales

    dX = np.empty_like(dOut)
    dX[:, 0::2] = dX1
    dX[:, 1::2] = dX2
    return dX

def train_xpos_component(seq_len=10, d_model=16, epochs=10000, learning_rate=0.5):
    np.random.seed(42)
    # Give inputs stronger values so model learns
    X = np.random.randn(seq_len, d_model)
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1

    cos_theta, sin_theta = get_rope_cos_sin(seq_len, d_model)
    scales = get_xpos_scale_base(seq_len, d_model)

    target_scores = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if i >= j:
                target_scores[i, j] = np.exp(-0.5 * (i - j))

    for epoch in range(epochs):
        Q_raw = np.dot(X, W_q)
        K_raw = np.dot(X, W_k)

        Q_xpos = apply_xpos(Q_raw, cos_theta, sin_theta, scales, is_query=True)
        K_xpos = apply_xpos(K_raw, cos_theta, sin_theta, scales, is_query=False)

        scores = np.dot(Q_xpos, K_xpos.T) / np.sqrt(d_model)

        loss = np.mean(0.5 * (scores - target_scores)**2)

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

        dScores = (scores - target_scores) / (seq_len * seq_len)

        dQ_xpos = np.dot(dScores, K_xpos) / np.sqrt(d_model)
        dK_xpos = np.dot(dScores.T, Q_xpos) / np.sqrt(d_model)

        dQ_raw = apply_xpos_backward(dQ_xpos, cos_theta, sin_theta, scales, is_query=True)
        dK_raw = apply_xpos_backward(dK_xpos, cos_theta, sin_theta, scales, is_query=False)

        dW_q = np.dot(X.T, dQ_raw)
        dW_k = np.dot(X.T, dK_raw)

        W_q -= learning_rate * dW_q
        W_k -= learning_rate * dW_k

    return W_q, W_k, scores

def main():
    parser = argparse.ArgumentParser(description="Train an XPOS component on synthetic data.")
    parser.add_argument("--d_model", type=int, default=16, help="Dimension of the model.")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate.")
    args = parser.parse_args()

    print(f"Training XPOS Component with seq_len={args.seq_len}, d_model={args.d_model}, epochs={args.epochs}, lr={args.lr}")
    W_q, W_k, final_scores = train_xpos_component(args.seq_len, args.d_model, args.epochs, args.lr)

    print("\nTraining Complete.")
    print("Final Scores (approximate target pattern):")
    np.set_printoptions(precision=2, suppress=True)
    print(final_scores)

if __name__ == "__main__":
    main()
