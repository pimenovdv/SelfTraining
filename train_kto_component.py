import numpy as np

def kto_loss(pi_logps, ref_logps, is_good, beta=0.1):
    """
    Kahneman-Tversky Optimization (KTO) Loss Component.

    Args:
        pi_logps: Log probabilities from the policy model. (batch_size,)
        ref_logps: Log probabilities from the reference model. (batch_size,)
        is_good: Boolean array indicating if the generation is "good" (1) or "bad" (0). (batch_size,)
        beta: KL penalty coefficient.

    Returns:
        loss: Scalar loss value.
    """
    # Calculate log ratio: log(pi / ref) = log pi - log ref
    log_ratio = pi_logps - ref_logps

    # In KTO, we apply different transformations based on whether the output is "good" or "bad"
    # To match standard implementations, we define ideal reward signals and use a logistic loss
    # Since we don't have pairs, we use the value relative to a reference point.
    # In full KTO, there's a KL term matching the unchosen/chosen distributions.
    # A simplified version:
    # If good: we want log_ratio to be large. Loss = -sigmoid(beta * (log_ratio - z_ref))
    # If bad: we want log_ratio to be small. Loss = -sigmoid(beta * (z_ref - log_ratio))
    # Here z_ref is the expected log ratio (KL). We will assume z_ref = 0 for simplicity.

    z_ref = 0.0 # simplified

    losses = []
    for i in range(len(is_good)):
        if is_good[i]:
            # Maximize log_ratio -> minimize -log(sigmoid(beta * log_ratio))
            loss_i = -np.log(1 / (1 + np.exp(-beta * (log_ratio[i] - z_ref))))
        else:
            # Minimize log_ratio -> minimize -log(sigmoid(beta * (-log_ratio)))
            loss_i = -np.log(1 / (1 + np.exp(-beta * (z_ref - log_ratio[i]))))
        losses.append(loss_i)

    return np.mean(losses)

def test_kto_component():
    np.random.seed(42)

    batch_size = 4
    pi_logps = np.array([-1.0, -2.0, -1.5, -0.5])
    ref_logps = np.array([-1.2, -1.8, -1.0, -0.8])
    is_good = np.array([1, 0, 1, 0])

    loss = kto_loss(pi_logps, ref_logps, is_good)
    print(f"KTO Loss: {loss:.4f}")
    assert loss > 0

if __name__ == "__main__":
    test_kto_component()
