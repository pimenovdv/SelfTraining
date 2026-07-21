import numpy as np
import argparse

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def layernorm_forward(X, gamma, beta, eps=1e-5):
    mu = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    X_hat = (X - mu) / std
    out = gamma * X_hat + beta
    return out, X_hat, mu, var, std

def layernorm_backward(dOut, X_hat, std, X, mu, gamma, eps=1e-5):
    N, D = X.shape
    dGamma = np.sum(dOut * X_hat, axis=0, keepdims=True)
    dBeta = np.sum(dOut, axis=0, keepdims=True)

    dX_hat = dOut * gamma
    dVar = np.sum(dX_hat * (X - mu) * -0.5 * (std**2)**(-1.5), axis=-1, keepdims=True)
    dMu = np.sum(dX_hat * -1 / std, axis=-1, keepdims=True) + dVar * np.mean(-2 * (X - mu), axis=-1, keepdims=True)

    dX = dX_hat / std + dVar * 2 * (X - mu) / D + dMu / D
    return dX, dGamma, dBeta

def train_multihead_transformer_block(X, y, d_model, num_heads, d_ff, epochs, learning_rate):
    seq_len = X.shape[0]
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_k = d_model // num_heads

    np.random.seed(42)
    # LayerNorm 1
    gamma1 = np.ones((1, d_model))
    beta1 = np.zeros((1, d_model))

    # Multi-Head Attention
    W_Q = np.random.randn(d_model, d_model) * 0.1
    W_K = np.random.randn(d_model, d_model) * 0.1
    W_V = np.random.randn(d_model, d_model) * 0.1
    W_O = np.random.randn(d_model, d_model) * 0.1

    # LayerNorm 2
    gamma2 = np.ones((1, d_model))
    beta2 = np.zeros((1, d_model))

    # FFN
    W1 = np.random.randn(d_model, d_ff) * 0.1
    b1 = np.zeros((1, d_ff))
    W2 = np.random.randn(d_ff, d_model) * 0.1
    b2 = np.zeros((1, d_model))

    for epoch in range(epochs):
        # --- FORWARD PASS ---
        # Pre-LN 1
        ln1_out, X_hat1, mu1, var1, std1 = layernorm_forward(X, gamma1, beta1)

        # Multi-Head Attention
        Q = np.dot(ln1_out, W_Q)
        K = np.dot(ln1_out, W_K)
        V = np.dot(ln1_out, W_V)

        # Split into heads (seq_len, num_heads, d_k)
        Q_split = Q.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
        K_split = K.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
        V_split = V.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

        # Scaled dot-product attention
        scores = np.matmul(Q_split, K_split.transpose(0, 2, 1)) / np.sqrt(d_k)
        attention_weights = softmax(scores)

        head_outputs = np.matmul(attention_weights, V_split)

        # Concatenate heads
        concat_outputs = head_outputs.transpose(1, 0, 2).reshape(seq_len, d_model)

        # Final linear layer
        attn_out = np.dot(concat_outputs, W_O)

        # Residual 1
        res1_out = X + attn_out

        # Pre-LN 2
        ln2_out, X_hat2, mu2, var2, std2 = layernorm_forward(res1_out, gamma2, beta2)

        # FFN
        ffn_z1 = np.dot(ln2_out, W1) + b1
        ffn_a1 = relu(ffn_z1)
        ffn_z2 = np.dot(ffn_a1, W2) + b2

        # Residual 2
        output = res1_out + ffn_z2

        # Loss
        loss = np.mean(0.5 * (output - y)**2)

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

        # --- BACKWARD PASS ---
        dOutput = (output - y) / (seq_len * d_model)

        # dResidual 2
        dRes1_out = dOutput.copy()
        dFfn_z2 = dOutput.copy()

        # FFN backward
        dW2 = np.dot(ffn_a1.T, dFfn_z2)
        db2 = np.sum(dFfn_z2, axis=0, keepdims=True)

        dFfn_a1 = np.dot(dFfn_z2, W2.T)
        dFfn_z1 = dFfn_a1 * relu_derivative(ffn_z1)

        dW1 = np.dot(ln2_out.T, dFfn_z1)
        db1 = np.sum(dFfn_z1, axis=0, keepdims=True)

        dLn2_out = np.dot(dFfn_z1, W1.T)

        # LayerNorm 2 backward
        dRes1_out_from_ln2, dGamma2, dBeta2 = layernorm_backward(
            dLn2_out, X_hat2, std2, res1_out, mu2, gamma2
        )

        # Combine gradients for res1_out
        dRes1_out_total = dRes1_out + dRes1_out_from_ln2

        # dResidual 1
        dX_from_res1 = dRes1_out_total.copy()
        dAttn_out = dRes1_out_total.copy()

        # Multi-Head Attention backward
        dW_O = np.dot(concat_outputs.T, dAttn_out)
        dConcat_outputs = np.dot(dAttn_out, W_O.T)

        # Split dConcat_outputs back to heads
        dHead_outputs = dConcat_outputs.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

        # Gradients for V and attention weights
        dV_split = np.matmul(attention_weights.transpose(0, 2, 1), dHead_outputs)
        dAttention_weights = np.matmul(dHead_outputs, V_split.transpose(0, 2, 1))

        # Gradient of softmax scores
        dScores = attention_weights * (dAttention_weights - np.sum(attention_weights * dAttention_weights, axis=-1, keepdims=True))

        # Gradients for Q and K
        dScores_scaled = dScores / np.sqrt(d_k)
        dQ_split = np.matmul(dScores_scaled, K_split)
        dK_split = np.matmul(dScores_scaled.transpose(0, 2, 1), Q_split)

        # Concatenate gradients for Q, K, V
        dQ = dQ_split.transpose(1, 0, 2).reshape(seq_len, d_model)
        dK = dK_split.transpose(1, 0, 2).reshape(seq_len, d_model)
        dV = dV_split.transpose(1, 0, 2).reshape(seq_len, d_model)

        # Gradients for W_Q, W_K, W_V
        dW_Q = np.dot(ln1_out.T, dQ)
        dW_K = np.dot(ln1_out.T, dK)
        dW_V = np.dot(ln1_out.T, dV)

        dLn1_out = np.dot(dQ, W_Q.T) + np.dot(dK, W_K.T) + np.dot(dV, W_V.T)

        # LayerNorm 1 backward
        dX_from_ln1, dGamma1, dBeta1 = layernorm_backward(
            dLn1_out, X_hat1, std1, X, mu1, gamma1
        )

        # Combine gradients for X
        dX_total = dX_from_res1 + dX_from_ln1

        # Update weights
        gamma1 -= learning_rate * dGamma1
        beta1 -= learning_rate * dBeta1
        gamma2 -= learning_rate * dGamma2
        beta2 -= learning_rate * dBeta2

        W_Q -= learning_rate * dW_Q
        W_K -= learning_rate * dW_K
        W_V -= learning_rate * dW_V
        W_O -= learning_rate * dW_O

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    return output

def main():
    parser = argparse.ArgumentParser(description="Train a Multi-Head Transformer Block component on synthetic data.")
    parser.add_argument("--d_model", type=int, default=4, help="Dimension of the model.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=8, help="Dimension of feed forward network hidden layer.")
    parser.add_argument("--epochs", type=int, default=20000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    args = parser.parse_args()

    # Synthetic Dataset
    X = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    y = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.5, 0.5],
        [1.0, 0.0, 1.0, 0.0]
    ])

    print(f"Training Multi-Head Transformer Block with d_model={args.d_model}, num_heads={args.num_heads}, d_ff={args.d_ff}, epochs={args.epochs}, lr={args.lr}")

    predictions = train_multihead_transformer_block(X, y, args.d_model, args.num_heads, args.d_ff, args.epochs, args.lr)

    print("\nTraining Complete.")
    print("Final Predictions:")
    print(predictions)
    print("Target:")
    print(y)

if __name__ == "__main__":
    main()
