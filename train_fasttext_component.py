import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_fasttext_component(epochs, lr):
    # Vocabulary size (words + subwords)
    V = 10
    # Embedding dimension
    d = 4

    np.random.seed(42)
    # Target embeddings (for words/subwords)
    W = np.random.randn(V, d) * 0.1
    # Context embeddings
    C = np.random.randn(V, d) * 0.1

    # Synthetic pairs: (target_components, context, label)
    # Target word is represented as a bag of n-grams/subwords
    training_data = [
        ([0, 5], 1, 1),
        ([0, 5], 2, 1),
        ([0, 5], 3, 0),
        ([0, 5], 4, 0),

        ([1, 6], 0, 1),
        ([1, 6], 2, 1),
        ([1, 6], 4, 0),
        ([1, 6], 3, 0)
    ]

    for epoch in range(epochs):
        total_loss = 0
        for target_components, context, label in training_data:
            # Word representation is sum of its subwords
            v_c = np.sum([W[idx] for idx in target_components], axis=0)
            u_o = C[context]

            score = np.dot(v_c, u_o)
            pred = sigmoid(score)

            eps = 1e-8
            loss = - (label * np.log(pred + eps) + (1 - label) * np.log(1 - pred + eps))
            total_loss += loss

            d_score = pred - label

            d_vc = d_score * u_o
            d_uo = d_score * v_c

            # Update all components of the target word
            for idx in target_components:
                W[idx] -= lr * d_vc
            C[context] -= lr * d_uo

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    success = total_loss < 0.5
    return success

if __name__ == "__main__":
    train_fasttext_component(1000, 0.1)
