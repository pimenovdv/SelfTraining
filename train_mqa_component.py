import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class MultiQueryAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # In MQA, there are multiple query heads but only a single key and value head
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, self.d_k) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, self.d_k) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Only 1 key/value head: (batch_size, seq_len, d_k) -> (batch_size, 1, seq_len, d_k)
        K = np.dot(x, self.W_k).reshape(batch_size, 1, seq_len, self.d_k)
        V = np.dot(x, self.W_v).reshape(batch_size, 1, seq_len, self.d_k)

        # Q: (batch_size, num_heads, seq_len, d_k)
        # K.transpose: (batch_size, 1, d_k, seq_len)
        # scores: (batch_size, num_heads, seq_len, seq_len)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        attn = softmax(scores, axis=-1)

        # context: (batch_size, num_heads, seq_len, d_k)
        context = np.matmul(attn, V)

        # (batch_size, seq_len, d_model)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        output = np.dot(context, self.W_o)
        return output

def test_mqa():
    np.random.seed(42)
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8

    x = np.random.randn(batch_size, seq_len, d_model)
    mqa = MultiQueryAttention(d_model, num_heads)

    output = mqa.forward(x)

    assert output.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("Multi-Query Attention (MQA) component forward pass successful.")

if __name__ == "__main__":
    test_mqa()
