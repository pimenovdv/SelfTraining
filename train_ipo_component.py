"""
Identity Preference Optimization (IPO) Component

This script implements and tests a simplified version of Identity Preference Optimization (IPO).
IPO is a preference learning algorithm related to Direct Preference Optimization (DPO),
but instead of using a cross-entropy loss, it defines the objective in terms of the
root of a quadratic loss, which theoretically avoids the over-fitting issues of DPO
when the preference dataset is deterministic or near-deterministic.

The IPO loss function is defined as:
L_IPO = E_{(x, y_w, y_l) ~ D} [ ( h_pi_theta(x, y_w) - h_pi_theta(x, y_l) - 1/(2 * tau) )^2 ]
where h_pi_theta(x, y) = log(pi_theta(y|x)) - log(pi_ref(y|x))
and tau is the temperature scaling parameter.
"""

import numpy as np

def ipo_loss(log_pi_theta_w, log_pi_theta_l, log_pi_ref_w, log_pi_ref_l, tau=0.1):
    """
    Computes Identity Preference Optimization (IPO) loss.

    Args:
        log_pi_theta_w: log probabilities of chosen actions under current policy pi_theta
        log_pi_theta_l: log probabilities of rejected actions under current policy pi_theta
        log_pi_ref_w: log probabilities of chosen actions under reference policy pi_ref
        log_pi_ref_l: log probabilities of rejected actions under reference policy pi_ref
        tau: temperature parameter

    Returns:
        IPO loss (scalar)
    """
    # h(x, y) = log_pi_theta - log_pi_ref
    h_w = log_pi_theta_w - log_pi_ref_w
    h_l = log_pi_theta_l - log_pi_ref_l

    # Margin
    margin = h_w - h_l

    # Target margin is 1 / (2 * tau)
    target = 1.0 / (2.0 * tau)

    # IPO loss is mean squared error to target margin
    loss = np.mean((margin - target)**2)
    return loss

def test_component():
    print("Testing Identity Preference Optimization (IPO) Component...")
    np.random.seed(42)

    # Simulate a batch of 4 examples
    batch_size = 4

    # Random log probabilities
    log_pi_theta_w = np.random.randn(batch_size)
    log_pi_theta_l = np.random.randn(batch_size) - 0.5 # rejected slightly worse

    log_pi_ref_w = np.random.randn(batch_size)
    log_pi_ref_l = np.random.randn(batch_size) - 0.5

    tau = 0.1

    loss = ipo_loss(log_pi_theta_w, log_pi_theta_l, log_pi_ref_w, log_pi_ref_l, tau)
    print(f"Calculated IPO Loss (tau={tau}): {loss:.4f}")

    # Check that if margin exactly matches 1/(2*tau), loss is 0
    target = 1.0 / (2.0 * tau)
    perfect_log_pi_theta_w = log_pi_ref_w + target
    perfect_log_pi_theta_l = log_pi_ref_l

    perfect_loss = ipo_loss(perfect_log_pi_theta_w, perfect_log_pi_theta_l, log_pi_ref_w, log_pi_ref_l, tau)
    print(f"Perfect Margin IPO Loss (expected 0.0): {perfect_loss:.4f}")

if __name__ == "__main__":
    test_component()
