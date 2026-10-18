import numpy as np
import argparse
import os

def info_nce_loss(query, positive_key, negative_keys, temperature=0.1):
    # Normalize embeddings
    query = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-8)
    positive_key = positive_key / (np.linalg.norm(positive_key, axis=-1, keepdims=True) + 1e-8)
    negative_keys = negative_keys / (np.linalg.norm(negative_keys, axis=-1, keepdims=True) + 1e-8)

    # Cosine similarities
    pos_sim = np.sum(query * positive_key, axis=-1)
    neg_sim = np.dot(query, negative_keys.T)

    pos_term = np.exp(pos_sim / temperature)
    neg_terms = np.exp(neg_sim / temperature)

    loss = -np.log(pos_term / (pos_term + np.sum(neg_terms, axis=-1)))
    return np.mean(loss)

def test_info_nce_loss():
    np.random.seed(42)
    query = np.random.randn(2, 64)
    positive_key = query + np.random.randn(2, 64) * 0.1 # Very similar
    negative_keys = np.random.randn(10, 64) # Different

    loss = info_nce_loss(query, positive_key, negative_keys)
    print(f"InfoNCE Loss: {loss:.4f}")
    assert loss < 1.0, "Loss should be small for similar positive keys"

    # Negative test
    positive_key_bad = np.random.randn(2, 64) # Random
    loss_bad = info_nce_loss(query, positive_key_bad, negative_keys)
    print(f"InfoNCE Loss (bad positive): {loss_bad:.4f}")
    assert loss_bad > loss, "Loss should be larger for dissimilar positive keys"

def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    print("Training InfoNCE Loss Component")
    test_info_nce_loss()
    print("Training Complete.")

if __name__ == "__main__":
    main()
