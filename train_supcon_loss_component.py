import numpy as np
import argparse

def supcon_loss(features, labels, temperature=0.1):
    # Normalize features
    features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)

    # Compute similarity matrix
    similarity_matrix = np.dot(features, features.T) / temperature

    batch_size = features.shape[0]

    # Create mask of positive pairs (same class)
    labels = labels.reshape(-1, 1)
    mask = (labels == labels.T).astype(float)

    # Mask out self-contrast
    logits_mask = np.ones((batch_size, batch_size)) - np.eye(batch_size)
    mask = mask * logits_mask

    # Compute log probabilities
    # For numerical stability
    max_sim = np.max(similarity_matrix, axis=1, keepdims=True)
    exp_sim = np.exp(similarity_matrix - max_sim) * logits_mask
    log_prob = similarity_matrix - max_sim - np.log(np.sum(exp_sim, axis=1, keepdims=True) + 1e-8)

    # Compute mean of log-likelihood over positive
    mask_sum = np.sum(mask, axis=1)
    mask_sum[mask_sum == 0] = 1 # Avoid division by zero

    mean_log_prob_pos = np.sum(mask * log_prob, axis=1) / mask_sum

    loss = -np.mean(mean_log_prob_pos)
    return loss

def test_supcon_loss():
    np.random.seed(42)
    # Create 4 samples: 2 of class 0, 2 of class 1
    features = np.array([
        [1.0, 0.1],
        [0.9, 0.2],
        [-1.0, 0.1],
        [-0.9, 0.2]
    ])
    labels = np.array([0, 0, 1, 1])

    loss = supcon_loss(features, labels)
    print(f"SupCon Loss (clustered): {loss:.4f}")
    assert loss < 1.0, "Loss should be small when classes are clustered"

    # Mixed up features
    features_bad = np.array([
        [1.0, 0.1],
        [-1.0, 0.1],
        [1.0, 0.1],
        [-1.0, 0.1]
    ])
    loss_bad = supcon_loss(features_bad, labels)
    print(f"SupCon Loss (mixed): {loss_bad:.4f}")
    assert loss_bad > loss, "Loss should be larger when classes are mixed"

def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    print("Training SupCon Loss Component")
    test_supcon_loss()
    print("Training Complete.")

if __name__ == "__main__":
    main()
