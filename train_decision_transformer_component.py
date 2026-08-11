import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class CausalSelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        # Using a single head for simplicity
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

    def forward(self, x):
        self.x = x
        self.q = np.dot(x, self.W_q)
        self.k = np.dot(x, self.W_k)
        self.v = np.dot(x, self.W_v)

        # q, k, v: (batch_size, seq_len, d_model)
        # scores: (batch_size, seq_len, seq_len)
        self.scores = np.matmul(self.q, self.k.transpose(0, 2, 1)) / np.sqrt(self.d_model)

        # Causal mask
        seq_len = x.shape[1]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        self.scores = np.where(mask == 1, -1e9, self.scores)

        self.attn_weights = softmax(self.scores, axis=-1)
        self.out = np.matmul(self.attn_weights, self.v)
        return self.out

    def backward(self, dout, lr=0.01):
        # dout: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = dout.shape

        dV = np.matmul(self.attn_weights.transpose(0, 2, 1), dout) # (batch, seq, d_model)
        self.dW_v = np.tensordot(self.x, dV, axes=([0, 1], [0, 1]))

        dAttn = np.matmul(dout, self.v.transpose(0, 2, 1)) # (batch, seq, seq)

        # gradient through softmax
        # dScores = attn_weights * (dAttn - sum(attn_weights * dAttn))
        sum_dAttn_attn = np.sum(dAttn * self.attn_weights, axis=-1, keepdims=True)
        dScores = self.attn_weights * (dAttn - sum_dAttn_attn)

        # Apply mask gradient (gradients for masked positions should be 0, they already are because attn_weights is ~0)
        dScores /= np.sqrt(self.d_model)

        dQ = np.matmul(dScores, self.k)
        dK = np.matmul(dScores.transpose(0, 2, 1), self.q)

        self.dW_q = np.tensordot(self.x, dQ, axes=([0, 1], [0, 1]))
        self.dW_k = np.tensordot(self.x, dK, axes=([0, 1], [0, 1]))

        dX = np.dot(dQ, self.W_q.T) + np.dot(dK, self.W_k.T) + np.dot(dV, self.W_v.T)

        # Update weights
        self.dW_q = np.clip(self.dW_q, -1.0, 1.0)
        self.dW_k = np.clip(self.dW_k, -1.0, 1.0)
        self.dW_v = np.clip(self.dW_v, -1.0, 1.0)

        self.W_q -= lr * self.dW_q
        self.W_k -= lr * self.dW_k
        self.W_v -= lr * self.dW_v

        return dX

class DecisionTransformerComponent:
    def __init__(self, state_dim, act_dim, rtg_dim, d_model, seq_len):
        self.d_model = d_model
        # Projections
        self.W_s = np.random.randn(state_dim, d_model) * 0.1
        self.W_a = np.random.randn(act_dim, d_model) * 0.1
        self.W_r = np.random.randn(rtg_dim, d_model) * 0.1

        # Timestep embeddings
        self.pos_emb = np.random.randn(seq_len, d_model) * 0.02

        # Causal Attention
        self.attn = CausalSelfAttention(d_model)

        # FFN
        self.W1 = np.random.randn(d_model, d_model * 4) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_model * 4)
        self.W2 = np.random.randn(d_model * 4, d_model) * np.sqrt(2.0 / (d_model * 4))
        self.b2 = np.zeros(d_model)

        # Action Predictor
        self.W_out = np.random.randn(d_model, act_dim) * np.sqrt(2.0 / d_model)
        self.b_out = np.zeros(act_dim)

    def forward(self, s, a_prev, rtg):
        # s: (batch, seq, state_dim)
        # a_prev: (batch, seq, act_dim)
        # rtg: (batch, seq, rtg_dim)

        self.s = s
        self.a_prev = a_prev
        self.rtg = rtg

        self.s_emb = np.dot(s, self.W_s)
        self.a_emb = np.dot(a_prev, self.W_a)
        self.r_emb = np.dot(rtg, self.W_r)

        # Combine embeddings (simple addition)
        self.x = self.s_emb + self.a_emb + self.r_emb + self.pos_emb

        # Attention
        self.attn_out = self.attn.forward(self.x)

        # Residual 1
        self.res1 = self.x + self.attn_out

        # FFN
        self.ffn1 = np.dot(self.res1, self.W1) + self.b1
        self.ffn1_relu = relu(self.ffn1)
        self.ffn2 = np.dot(self.ffn1_relu, self.W2) + self.b2

        # Residual 2
        self.res2 = self.res1 + self.ffn2

        # Action Predictor
        self.action_pred = np.dot(self.res2, self.W_out) + self.b_out
        return self.action_pred

    def backward(self, dout, lr=0.01):
        # dout: (batch_size, seq_len, act_dim) (gradients of MSE loss)

        self.dW_out = np.tensordot(self.res2, dout, axes=([0, 1], [0, 1]))
        self.db_out = np.sum(dout, axis=(0, 1))

        dRes2 = np.dot(dout, self.W_out.T)

        dFfn2 = dRes2
        self.dW2 = np.tensordot(self.ffn1_relu, dFfn2, axes=([0, 1], [0, 1]))
        self.db2 = np.sum(dFfn2, axis=(0, 1))

        dFfn1_relu = np.dot(dFfn2, self.W2.T)
        dFfn1 = dFfn1_relu * relu_derivative(self.ffn1)

        self.dW1 = np.tensordot(self.res1, dFfn1, axes=([0, 1], [0, 1]))
        self.db1 = np.sum(dFfn1, axis=(0, 1))

        dRes1 = dRes2 + np.dot(dFfn1, self.W1.T)

        dAttn_out = dRes1
        dX_attn = self.attn.backward(dAttn_out, lr)

        dX = dRes1 + dX_attn

        # Gradients for embeddings
        self.dPos_emb = np.sum(dX, axis=0) # (seq_len, d_model)

        self.dW_s = np.tensordot(self.s, dX, axes=([0, 1], [0, 1]))
        self.dW_a = np.tensordot(self.a_prev, dX, axes=([0, 1], [0, 1]))
        self.dW_r = np.tensordot(self.rtg, dX, axes=([0, 1], [0, 1]))

        # Updates
        self.dW_out = np.clip(self.dW_out, -1.0, 1.0)
        self.db_out = np.clip(self.db_out, -1.0, 1.0)
        self.dW2 = np.clip(self.dW2, -1.0, 1.0)
        self.db2 = np.clip(self.db2, -1.0, 1.0)
        self.dW1 = np.clip(self.dW1, -1.0, 1.0)
        self.db1 = np.clip(self.db1, -1.0, 1.0)
        self.dPos_emb = np.clip(self.dPos_emb, -1.0, 1.0)
        self.dW_s = np.clip(self.dW_s, -1.0, 1.0)
        self.dW_a = np.clip(self.dW_a, -1.0, 1.0)
        self.dW_r = np.clip(self.dW_r, -1.0, 1.0)

        self.W_out -= lr * self.dW_out
        self.b_out -= lr * self.db_out

        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1

        self.pos_emb -= lr * self.dPos_emb
        self.W_s -= lr * self.dW_s
        self.W_a -= lr * self.dW_a
        self.W_r -= lr * self.dW_r

        return dX


def generate_offline_trajectories(num_traj=100, seq_len=10):
    # Let's create a simple 1D continuous task: moving to position 0 from an initial random position.
    # State: x position.
    # Action: velocity (x_new = x + action).
    # Reward: -|x|.
    # Expert policy: action = -0.5 * x.
    # Return-to-go (RTG) will be computed from expert trajectories.

    S = np.zeros((num_traj, seq_len, 1))
    A = np.zeros((num_traj, seq_len, 1))
    R = np.zeros((num_traj, seq_len, 1))
    RTG = np.zeros((num_traj, seq_len, 1))

    for i in range(num_traj):
        x = np.random.uniform(-10, 10)
        rewards = []
        for t in range(seq_len):
            a = -0.5 * x + np.random.normal(0, 0.1) # Expert with slight noise
            r = -np.abs(x)

            S[i, t, 0] = x
            A[i, t, 0] = a
            R[i, t, 0] = r

            x = x + a

        # Compute RTG
        rtg = 0
        for t in reversed(range(seq_len)):
            rtg += R[i, t, 0]
            RTG[i, t, 0] = rtg

    return S, A, RTG

if __name__ == "__main__":
    print("Generating offline expert trajectories...")
    seq_len = 10
    S, A, RTG = generate_offline_trajectories(num_traj=200, seq_len=seq_len)

    # We need a_prev. a_0 is 0.
    A_prev = np.zeros_like(A)
    A_prev[:, 1:, :] = A[:, :-1, :]

    dt = DecisionTransformerComponent(state_dim=1, act_dim=1, rtg_dim=1, d_model=16, seq_len=seq_len)

    epochs = 200
    lr = 0.005
    batch_size = 32

    print("Training Decision Transformer on offline trajectories...")
    for epoch in range(epochs):
        indices = np.random.permutation(len(S))
        S_batch = S[indices[:batch_size]]
        A_prev_batch = A_prev[indices[:batch_size]]
        RTG_batch = RTG[indices[:batch_size]]
        A_target_batch = A[indices[:batch_size]]

        # Forward
        action_pred = dt.forward(S_batch, A_prev_batch, RTG_batch)

        # MSE Loss
        loss = np.mean((action_pred - A_target_batch) ** 2)

        # Backward
        dout = 2.0 * (action_pred - A_target_batch) / (batch_size * seq_len)
        dout = np.clip(dout, -1.0, 1.0)
        dt.backward(dout, lr=lr)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

    print(f"Final Loss: {loss:.4f}")
    assert loss < 0.5, "Decision Transformer failed to learn the expert policy!"
    print("Decision Transformer component successfully modeled offline RL via sequence modeling.")
