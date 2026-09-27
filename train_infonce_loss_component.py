import numpy as np

def infonce_loss(features, temperature=0.1):
    """
    Computes InfoNCE (Information Noise-Contrastive Estimation) Loss.
    Assumes `features` is a numpy array of shape (N, 2, D), where:
    - N is the batch size (number of positive pairs)
    - 2 represents the pairs (anchor and positive)
    - D is the embedding dimension
    Returns the average InfoNCE loss over the batch.
    """
    N = features.shape[0]

    # Split into anchors and positives
    anchors = features[:, 0, :]   # Shape: (N, D)
    positives = features[:, 1, :] # Shape: (N, D)

    # Normalize features to compute cosine similarity directly via dot product
    anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
    positives = positives / np.linalg.norm(positives, axis=1, keepdims=True)

    # Compute similarity scores: dot product between all anchors and all positives
    # Shape of sim_scores: (N, N)
    sim_scores = np.dot(anchors, positives.T) / temperature

    # Exponentiate the similarity scores
    exp_sim = np.exp(sim_scores)

    # Compute the denominator for each anchor (sum over all columns in the row)
    sum_exp_sim = np.sum(exp_sim, axis=1)

    # The numerator is the exponentiated similarity of the positive pair (the diagonal)
    pos_exp_sim = np.diag(exp_sim)

    # Compute the loss: -log(numerator / denominator)
    losses = -np.log(pos_exp_sim / sum_exp_sim)

    # Average the loss over the batch
    loss = np.mean(losses)

    return loss

def test_infonce_loss_component():
    np.random.seed(42)

    N = 16 # Batch size
    D = 64 # Feature dimension

    # Generate random features for N pairs
    anchors = np.random.randn(N, D)
    positives = anchors + 0.1 * np.random.randn(N, D)

    features = np.stack([anchors, positives], axis=1)

    loss = infonce_loss(features, temperature=0.5)

    print(f"Computed InfoNCE Loss: {loss:.4f}")
    assert loss > 0, "Loss should be positive."

    # Test perfectly aligned positive pairs
    perfect_features = np.stack([anchors, anchors], axis=1)
    perfect_loss = infonce_loss(perfect_features, temperature=0.5)
    print(f"InfoNCE Loss for perfect alignment: {perfect_loss:.4f}")
    assert perfect_loss < loss, "Loss should be smaller for perfectly aligned pairs."

    # Test random, unaligned features
    random_positives = np.random.randn(N, D)
    random_features = np.stack([anchors, random_positives], axis=1)
    random_loss = infonce_loss(random_features, temperature=0.5)
    print(f"InfoNCE Loss for random pairs: {random_loss:.4f}")
    assert random_loss > loss, "Loss should be larger for unaligned/random pairs."

    print("InfoNCE Loss component test passed.")

if __name__ == "__main__":
    test_infonce_loss_component()
