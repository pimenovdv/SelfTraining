import numpy as np

def sparse_attention(Q, K, V, block_size=2):
    """
    Simulates a blocked sparse attention mechanism mathematically.
    Q, K, V shape: (batch_size, seq_len, d_k)
    Only blocks along the diagonal and immediate off-diagonals are computed.
    """
    batch_size, seq_len, d_k = Q.shape
    assert seq_len % block_size == 0

    num_blocks = seq_len // block_size
    output = np.zeros_like(Q)

    for b in range(batch_size):
        for i in range(num_blocks):
            # Compute attention only for local window (e.g. i-1 to i+1 blocks)
            start_j = max(0, i - 1)
            end_j = min(num_blocks, i + 2)

            q_block = Q[b, i*block_size : (i+1)*block_size, :]

            # Gather K and V for the local window
            k_local = K[b, start_j*block_size : end_j*block_size, :]
            v_local = V[b, start_j*block_size : end_j*block_size, :]

            # Scaled Dot-Product
            scores = np.dot(q_block, k_local.T) / np.sqrt(d_k)

            # Softmax
            max_scores = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            # Output
            out_block = np.dot(attn_weights, v_local)
            output[b, i*block_size : (i+1)*block_size, :] = out_block

    return output

def test_component():
    np.random.seed(42)
    batch_size = 2
    seq_len = 6
    d_k = 4
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    output = sparse_attention(Q, K, V, block_size=2)
    assert output.shape == (batch_size, seq_len, d_k)
    print("Sparse Attention Component successfully evaluated.")

if __name__ == "__main__":
    test_component()
