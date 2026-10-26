import numpy as np
import warnings

# Suppress overflow warnings in exponential/sigmoid for testing purposes
warnings.filterwarnings('ignore', category=RuntimeWarning)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class RestrictedBoltzmannMachine:
    def __init__(self, visible_units, hidden_units, learning_rate=0.01):
        """
        Initializes a Restricted Boltzmann Machine (RBM).

        Args:
            visible_units (int): Number of visible units (features).
            hidden_units (int): Number of hidden units.
            learning_rate (float): Learning rate for Contrastive Divergence.
        """
        self.v_dim = visible_units
        self.h_dim = hidden_units
        self.lr = learning_rate

        # Initialize weights and biases
        # Weights matrix of shape (visible_units, hidden_units)
        self.W = np.random.normal(scale=0.01, size=(self.v_dim, self.h_dim))
        self.v_bias = np.zeros(self.v_dim)
        self.h_bias = np.zeros(self.h_dim)

    def sample_hidden_given_visible(self, v):
        """
        Calculates the probability of hidden units given visible units and samples them.
        """
        h_prob = sigmoid(np.dot(v, self.W) + self.h_bias)
        # Sample states based on probabilities
        h_sample = np.random.binomial(1, h_prob)
        return h_prob, h_sample

    def sample_visible_given_hidden(self, h):
        """
        Calculates the probability of visible units given hidden units and samples them.
        """
        v_prob = sigmoid(np.dot(h, self.W.T) + self.v_bias)
        v_sample = np.random.binomial(1, v_prob)
        return v_prob, v_sample

    def contrastive_divergence(self, data, k=1):
        """
        Performs one step of Contrastive Divergence (CD-k) learning.

        Args:
            data (np.ndarray): Batch of training data, shape (batch_size, v_dim).
            k (int): Number of Gibbs sampling steps.

        Returns:
            float: Mean squared error for this batch.
        """
        num_samples = data.shape[0]

        # Positive Phase
        v0 = data
        h0_prob, h0_sample = self.sample_hidden_given_visible(v0)

        # Calculate positive associations (v_i * h_j for the data)
        pos_associations = np.dot(v0.T, h0_prob)

        # Negative Phase (Gibbs Sampling)
        v_k = v0
        h_k_sample = h0_sample
        for _ in range(k):
            v_k_prob, v_k = self.sample_visible_given_hidden(h_k_sample)
            h_k_prob, h_k_sample = self.sample_hidden_given_visible(v_k)

        # Calculate negative associations (v_i * h_j for the reconstruction)
        neg_associations = np.dot(v_k.T, h_k_prob)

        # Update weights and biases
        self.W += self.lr * (pos_associations - neg_associations) / num_samples
        self.v_bias += self.lr * np.mean(v0 - v_k, axis=0)
        self.h_bias += self.lr * np.mean(h0_prob - h_k_prob, axis=0)

        # Calculate reconstruction error
        error = np.mean(np.sum((v0 - v_k)**2, axis=1))
        return error

    def train(self, data, epochs=100, batch_size=10, k=1):
        """
        Trains the RBM on the provided data.
        """
        num_samples = data.shape[0]
        history = []

        for epoch in range(epochs):
            np.random.shuffle(data)
            epoch_errors = []

            for i in range(0, num_samples, batch_size):
                batch = data[i:i+batch_size]
                error = self.contrastive_divergence(batch, k=k)
                epoch_errors.append(error)

            mean_epoch_error = np.mean(epoch_errors)
            history.append(mean_epoch_error)

        return history

    def reconstruct(self, v):
        """Reconstructs the visible input."""
        h_prob, _ = self.sample_hidden_given_visible(v)
        v_prob, _ = self.sample_visible_given_hidden(h_prob)
        return v_prob

if __name__ == "__main__":
    print("Testing Restricted Boltzmann Machine (RBM) Component...")

    np.random.seed(42)

    # Generate synthetic binary dataset (e.g., small patterns)
    # Pattern 1: [1, 1, 1, 0, 0, 0]
    # Pattern 2: [0, 0, 0, 1, 1, 1]
    pattern1 = np.array([1, 1, 1, 0, 0, 0])
    pattern2 = np.array([0, 0, 0, 1, 1, 1])

    data = []
    for _ in range(50):
        # Add slight noise by occasionally flipping bits
        p1 = pattern1.copy()
        if np.random.rand() > 0.9: p1[np.random.randint(6)] ^= 1
        data.append(p1)

        p2 = pattern2.copy()
        if np.random.rand() > 0.9: p2[np.random.randint(6)] ^= 1
        data.append(p2)

    data = np.array(data)

    # Initialize and train RBM
    rbm = RestrictedBoltzmannMachine(visible_units=6, hidden_units=2, learning_rate=0.1)
    history = rbm.train(data, epochs=200, batch_size=10, k=1)

    final_error = history[-1]
    print(f"Final Reconstruction Error: {final_error:.4f}")

    # Test reconstruction on clean patterns
    rec_p1 = rbm.reconstruct(pattern1)
    rec_p2 = rbm.reconstruct(pattern2)

    print(f"Original Pattern 1: {pattern1}")
    print(f"Reconstructed Pattern 1 (Probs): {rec_p1.round(2)}")
    print(f"Original Pattern 2: {pattern2}")
    print(f"Reconstructed Pattern 2 (Probs): {rec_p2.round(2)}")

    # Verification
    # The reconstructed probabilities should be close to the original binary values (thresholding at 0.5)
    assert np.all((rec_p1 > 0.5).astype(int) == pattern1), "Failed to reconstruct Pattern 1."
    assert np.all((rec_p2 > 0.5).astype(int) == pattern2), "Failed to reconstruct Pattern 2."

    print("Restricted Boltzmann Machine component successfully verified!")
