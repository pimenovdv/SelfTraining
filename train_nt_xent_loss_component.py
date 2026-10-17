import numpy as np

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss for SimCLR.
    """
    N = z_i.shape[0]
    z = np.concatenate([z_i, z_j], axis=0) # Shape: (2N, D)

    # Normalize embeddings to lie on unit hypersphere
    z_norm = z / np.linalg.norm(z, axis=1, keepdims=True)

    # Cosine similarity matrix
    sim_matrix = np.dot(z_norm, z_norm.T) / temperature # Shape: (2N, 2N)

    # Mask out self-similarity
    np.fill_diagonal(sim_matrix, -np.inf)

    # Create labels: Positive pair for i is N+i, and for N+i is i
    labels = np.concatenate([np.arange(N, 2*N), np.arange(N)])

    # Softmax over similarities
    exp_sim = np.exp(sim_matrix)
    sum_exp_sim = np.sum(exp_sim, axis=1, keepdims=True)
    prob_matrix = exp_sim / sum_exp_sim

    # Compute the cross entropy loss
    loss = -np.mean(np.log(prob_matrix[np.arange(2*N), labels]))
    return loss

def main():
    # Simulate batch of representations
    np.random.seed(42)
    N = 4 # Batch size
    D = 16 # Embedding dimension

    # Positive pairs with high similarity
    z_i = np.random.randn(N, D)
    z_j = z_i + np.random.randn(N, D) * 0.1

    loss_positive = nt_xent_loss(z_i, z_j)
    print(f"NT-Xent Loss with high similarity (positive pairs): {loss_positive:.4f}")

    # Uncorrelated pairs with lower similarity
    z_j_random = np.random.randn(N, D)
    loss_random = nt_xent_loss(z_i, z_j_random)
    print(f"NT-Xent Loss with random pairing: {loss_random:.4f}")

    assert loss_positive < loss_random, "Loss should be lower when representations are similar"
    print("NT-Xent loss component verified successfully.")

if __name__ == "__main__":
    main()
