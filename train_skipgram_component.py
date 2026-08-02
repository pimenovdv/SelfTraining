import numpy as np
import os

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_skipgram_component(epochs, lr):
    # Vocabulary size
    V = 5
    # Embedding dimension
    d = 4

    np.random.seed(42)
    # Target embeddings
    W = np.random.randn(V, d) * 0.1
    # Context embeddings
    C = np.random.randn(V, d) * 0.1

    # Synthetic pairs (target, context, label (1=positive, 0=negative))
    # Context window target=0, true_context=1, true_context=2
    # Negative samples for target=0 are 3, 4
    training_data = [
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, 0),
        (0, 4, 0),

        (1, 0, 1),
        (1, 2, 1),
        (1, 4, 0),
        (1, 3, 0)
    ]

    for epoch in range(epochs):
        total_loss = 0
        for target, context, label in training_data:
            v_c = W[target]
            u_o = C[context]

            score = np.dot(v_c, u_o)
            pred = sigmoid(score)

            # Loss: - [y * log(p) + (1-y) * log(1-p)]
            # If label == 1: -log(pred), else -log(1-pred)
            eps = 1e-8
            loss = - (label * np.log(pred + eps) + (1 - label) * np.log(1 - pred + eps))
            total_loss += loss

            # Gradients
            # dL/d(score) = pred - label
            d_score = pred - label

            d_vc = d_score * u_o
            d_uo = d_score * v_c

            W[target] -= lr * d_vc
            C[context] -= lr * d_uo

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    success = total_loss < 0.5

    os.makedirs("docs", exist_ok=True)
    report_path = "docs/0077_train_skipgram_component.md"
    report_content = f"""# Experiment 0077: Train Skip-Gram Component

## Objective
Implement and mathematically model a Skip-Gram component with Negative Sampling, testing the hypothesis that word representations can be learned by maximizing the similarity between target words and their contexts while minimizing similarity with negative samples via manual backpropagation.

## Setup
*   **Script:** `train_skipgram_component.py`
*   **Data:** Synthetic word context pairs.
*   **Hyperparameters:** `epochs` = {epochs}, `learning_rate` = {lr}, `V` (vocab) = {V}, `d` (embed_dim) = {d}

## Execution
The script was executed to verify the mathematical formulation of Skip-Gram Negative Sampling and the manual backpropagation of gradients to update target and context embeddings.

## Results
*   **Status:** {"Success" if success else "Failed"}
*   **Final Loss:** {total_loss:.4f}
"""
    with open(report_path, "w") as f:
        f.write(report_content)

    return success

if __name__ == "__main__":
    train_skipgram_component(5000, 0.1)
