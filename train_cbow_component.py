import numpy as np
import os

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def train_cbow_component(epochs, lr):
    # Vocabulary size
    V = 5
    # Embedding dimension
    d = 4

    np.random.seed(42)
    # Context embeddings (input)
    C = np.random.randn(V, d) * 0.1
    # Target embeddings (output)
    W = np.random.randn(V, d) * 0.1

    # Synthetic data (context_indices, target_index)
    training_data = [
        ([1, 2], 0),
        ([0, 3], 1),
        ([0, 1], 2),
        ([1, 4], 3)
    ]

    for epoch in range(epochs):
        total_loss = 0
        for context, target in training_data:
            # Average context embeddings
            h = np.mean(C[context], axis=0) # shape (d,)

            # Scores for all words
            scores = np.dot(W, h) # shape (V,)

            # Predict probabilities
            probs = softmax(scores)

            # Cross-entropy loss
            loss = -np.log(probs[target] + 1e-8)
            total_loss += loss

            # Gradients
            # dL/dscores = probs - one_hot_target
            d_scores = probs.copy()
            d_scores[target] -= 1.0 # shape (V,)

            # dL/dW = d_scores outer h
            d_W = np.outer(d_scores, h) # shape (V, d)

            # dL/dh = W.T.dot(d_scores)
            d_h = np.dot(W.T, d_scores) # shape (d,)

            # dL/dC for context words
            d_C = d_h / len(context)

            W -= lr * d_W
            for c_idx in context:
                C[c_idx] -= lr * d_C

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    success = total_loss < 0.5

    os.makedirs("docs", exist_ok=True)
    report_path = "docs/0078_train_cbow_component.md"
    report_content = f"""# Experiment 0078: Train Continuous Bag of Words (CBOW) Component

## Objective
Implement and mathematically model a Continuous Bag of Words (CBOW) component, testing the hypothesis that word representations can be learned by predicting a target word from the average of its context word embeddings, utilizing manual backpropagation.

## Setup
*   **Script:** `train_cbow_component.py`
*   **Data:** Synthetic context-target word pairs.
*   **Hyperparameters:** `epochs` = {epochs}, `learning_rate` = {lr}, `V` (vocab) = {V}, `d` (embed_dim) = {d}

## Execution
The script was executed to verify the mathematical formulation of CBOW and the manual backpropagation of gradients to update context and target embeddings.

## Results
*   **Status:** {"Success" if success else "Failed"}
*   **Final Loss:** {total_loss:.4f}
"""
    with open(report_path, "w") as f:
        f.write(report_content)

    return success

if __name__ == "__main__":
    train_cbow_component(2000, 0.1)
