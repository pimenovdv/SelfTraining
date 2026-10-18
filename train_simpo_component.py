import numpy as np

def simpo_loss(chosen_log_probs, chosen_lengths, rejected_log_probs, rejected_lengths, beta=2.0, gamma=0.5):
    """
    Computes the SimPO (Simple Preference Optimization) loss.

    Args:
    chosen_log_probs: np.array of sum of log probabilities for chosen responses.
    chosen_lengths: np.array of lengths for chosen responses.
    rejected_log_probs: np.array of sum of log probabilities for rejected responses.
    rejected_lengths: np.array of lengths for rejected responses.
    beta: scaling factor.
    gamma: target reward margin.
    """
    # Length normalized rewards
    pi_theta_chosen = chosen_log_probs / chosen_lengths
    pi_theta_rejected = rejected_log_probs / rejected_lengths

    # Reward difference
    reward_diff = pi_theta_chosen - pi_theta_rejected

    # Margin-based contrastive loss
    # Using log-sigmoid: -log(sigmoid(beta * (reward_diff - gamma)))
    loss = -np.mean(np.log(1 / (1 + np.exp(-beta * (reward_diff - gamma)))))

    return loss, pi_theta_chosen, pi_theta_rejected

def test_simpo():
    print("Testing SimPO Component...")

    # Dummy data
    np.random.seed(42)
    batch_size = 4

    # Suppose chosen responses have slightly better log probs (less negative)
    chosen_log_probs = np.array([-10.5, -12.2, -8.9, -15.1])
    chosen_lengths = np.array([10, 12, 8, 15])

    rejected_log_probs = np.array([-14.2, -18.5, -12.4, -20.3])
    rejected_lengths = np.array([12, 14, 10, 18])

    loss, pi_chosen, pi_rejected = simpo_loss(
        chosen_log_probs, chosen_lengths,
        rejected_log_probs, rejected_lengths,
        beta=2.0, gamma=0.5
    )

    print(f"SimPO Loss: {loss:.4f}")
    print(f"Chosen Rewards: {pi_chosen}")
    print(f"Rejected Rewards: {pi_rejected}")

    if np.all(pi_chosen > pi_rejected):
        print("Test passed successfully: Chosen rewards are higher than rejected rewards.")
    else:
        print("Test failed: Rewards mismatch.")

if __name__ == "__main__":
    test_simpo()
