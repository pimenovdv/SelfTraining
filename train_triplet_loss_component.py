import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Computes the triplet loss.
    anchor, positive, negative: arrays of shape (batch_size, embedding_dim).
    """
    pos_dist = np.sum(np.square(anchor - positive), axis=1)
    neg_dist = np.sum(np.square(anchor - negative), axis=1)

    loss = np.maximum(pos_dist - neg_dist + margin, 0)
    return np.mean(loss)

def triplet_loss_gradient(anchor, positive, negative, margin=1.0):
    """
    Computes the gradients for the triplet loss w.r.t anchor, positive, and negative.
    """
    pos_dist = np.sum(np.square(anchor - positive), axis=1)
    neg_dist = np.sum(np.square(anchor - negative), axis=1)

    # Identify which triplets have a positive loss
    active = pos_dist - neg_dist + margin > 0
    active = active.reshape(-1, 1) # reshape for broadcasting

    N = anchor.shape[0]

    # Gradients
    # d(pos_dist) / d(anchor) = 2 * (anchor - positive)
    # d(neg_dist) / d(anchor) = 2 * (anchor - negative)
    # d(loss) / d(anchor) = d(pos_dist) - d(neg_dist)
    grad_anchor = 2 * (negative - positive) * active / N

    # d(loss) / d(positive) = -2 * (anchor - positive)
    grad_positive = -2 * (anchor - positive) * active / N

    # d(loss) / d(negative) = 2 * (anchor - negative)
    grad_negative = 2 * (anchor - negative) * active / N

    return grad_anchor, grad_positive, grad_negative

def test_triplet_loss():
    print("Testing Triplet Loss Component...")

    # Setup some dummy embeddings
    np.random.seed(42)
    batch_size = 32
    emb_dim = 16

    anchor = np.random.randn(batch_size, emb_dim)
    # Positive is close to anchor
    positive = anchor + 0.1 * np.random.randn(batch_size, emb_dim)
    # Negative is far from anchor
    negative = anchor + 2.0 * np.random.randn(batch_size, emb_dim)

    loss = triplet_loss(anchor, positive, negative, margin=1.0)
    print(f"Initial Loss (should be 0 or close to it): {loss:.4f}")

    # Now let's try a case where negative is closer
    negative_close = anchor + 0.05 * np.random.randn(batch_size, emb_dim)
    loss_high = triplet_loss(anchor, positive, negative_close, margin=1.0)
    print(f"High Loss (negative too close): {loss_high:.4f}")

    grad_a, grad_p, grad_n = triplet_loss_gradient(anchor, positive, negative_close, margin=1.0)
    print(f"Gradient magnitudes - Anchor: {np.linalg.norm(grad_a):.4f}, Pos: {np.linalg.norm(grad_p):.4f}, Neg: {np.linalg.norm(grad_n):.4f}")

    if loss_high > loss:
        print("Test passed successfully.")
    else:
        print("Test failed.")

if __name__ == "__main__":
    test_triplet_loss()
