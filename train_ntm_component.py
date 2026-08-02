import numpy as np
import os

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def cosine_similarity(k, M):
    norm_k = np.linalg.norm(k)
    norm_M = np.linalg.norm(M, axis=1)
    eps = 1e-8
    return np.dot(M, k) / ((norm_k * norm_M) + eps)

class NTMMemory:
    def __init__(self, N, d):
        self.N = N
        self.d = d
        self.M = np.random.randn(N, d) * 0.1

    def read(self, w):
        return np.dot(w, self.M)

    def write(self, w, e, a):
        M_tilde = self.M * (1 - np.outer(w, e))
        self.M = M_tilde + np.outer(w, a)

def train_ntm_component(epochs, lr):
    N, d = 4, 8

    np.random.seed(42)
    memory = NTMMemory(N, d)

    k1 = np.random.randn(d)
    v1 = np.random.randn(d)

    k2 = np.random.randn(d)
    v2 = np.random.randn(d)

    w_write_1 = np.array([1, 0, 0, 0], dtype=float)
    memory.write(w_write_1, np.ones(d), v1)

    w_write_2 = np.array([0, 1, 0, 0], dtype=float)
    memory.write(w_write_2, np.ones(d), v2)

    q1 = np.random.randn(d)
    q2 = np.random.randn(d)

    beta = 5.0

    for epoch in range(epochs):
        sim1 = cosine_similarity(q1, memory.M)
        w1 = softmax(beta * sim1)
        out1 = memory.read(w1)
        loss1 = np.mean((out1 - v1)**2)

        dout1 = 2 * (out1 - v1) / d
        dw1 = np.dot(memory.M, dout1)
        d_softmax_input1 = w1 * (dw1 - np.sum(w1 * dw1))
        d_sim1 = beta * d_softmax_input1

        norm_q1 = np.linalg.norm(q1)
        norm_M = np.linalg.norm(memory.M, axis=1)
        eps = 1e-8

        dq1 = np.zeros_like(q1)
        for i in range(N):
            dot_prod = np.dot(memory.M[i], q1)
            term1 = memory.M[i] / ((norm_M[i] * norm_q1) + eps)
            term2 = (dot_prod / ((norm_M[i] * norm_q1**3) + eps)) * q1
            dq1 += d_sim1[i] * (term1 - term2)

        q1 -= lr * dq1

        sim2 = cosine_similarity(q2, memory.M)
        w2 = softmax(beta * sim2)
        out2 = memory.read(w2)
        loss2 = np.mean((out2 - v2)**2)

        dout2 = 2 * (out2 - v2) / d
        dw2 = np.dot(memory.M, dout2)
        d_softmax_input2 = w2 * (dw2 - np.sum(w2 * dw2))
        d_sim2 = beta * d_softmax_input2

        norm_q2 = np.linalg.norm(q2)
        dq2 = np.zeros_like(q2)
        for i in range(N):
            dot_prod = np.dot(memory.M[i], q2)
            term1 = memory.M[i] / ((norm_M[i] * norm_q2) + eps)
            term2 = (dot_prod / ((norm_M[i] * norm_q2**3) + eps)) * q2
            dq2 += d_sim2[i] * (term1 - term2)

        q2 -= lr * dq2

        loss = loss1 + loss2
        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    success = loss < 0.01

    os.makedirs("docs", exist_ok=True)
    report_path = "docs/0076_train_ntm_component.md"
    report_content = f"""# Experiment 0076: Train Neural Turing Machine (NTM) Component

## Objective
Implement and mathematically model a Neural Turing Machine (NTM) component, testing the hypothesis that differentiable external memory can be addressed via content similarity and selectively read from and written to using backpropagation.

## Setup
*   **Script:** `train_ntm_component.py`
*   **Data:** Synthetic key-value retrieval task.
*   **Hyperparameters:** `epochs` = {epochs}, `learning_rate` = {lr}, `N` (memory slots) = {N}, `d` (memory dimension) = {d}

## Execution
The script was executed to verify the mathematical formulation of content-based addressing (cosine similarity + softmax) and the manual backpropagation of gradients through the memory read mechanism to optimize a query vector.

## Results
*   **Status:** {"Success" if success else "Failed"}
*   **Final Loss:** {loss:.4f}
"""
    with open(report_path, "w") as f:
        f.write(report_content)

    return success

if __name__ == "__main__":
    train_ntm_component(1000, 0.5)
