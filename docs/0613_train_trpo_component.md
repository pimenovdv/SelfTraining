# TRPO Component

**Script:** train_trpo_component.py

## Overview
Trust Region Policy Optimization (TRPO) is an advanced policy gradient reinforcement learning algorithm that mathematically optimizes policies by taking steps that are guaranteed to improve performance monotonically. It does this by enforcing a Kullback-Leibler (KL) divergence constraint on policy updates, ensuring the new policy doesn't deviate too much from the old one, thus maintaining a "trust region". This script implements TRPO mathematically, training an actor-critic model using conjugate gradient and line search to optimize a policy on a simple discrete environment.

## Mathematical Details
1.  **Actor-Critic Architecture:** The model consists of a shared neural network representation, branching into an actor (policy $\pi_\theta(a|s)$) and a critic (value function $V(s)$).
2.  **Advantage Estimation:** Advantages $A(s_t, a_t)$ are computed as the difference between the discounted returns and the baseline value estimates.
3.  **Surrogate Objective:** TRPO maximizes the expected surrogate objective:
    $L(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right]$
4.  **KL Divergence Constraint:** The optimization is subject to a constraint on the maximum average KL divergence between the old and new policies:
    $\hat{\mathbb{E}}_t [KL[\pi_{\theta_{old}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]] \le \delta$
5.  **Fisher Information Matrix (FIM) & Conjugate Gradient:** Because computing the full inverse of the FIM is computationally expensive, TRPO uses the conjugate gradient algorithm to approximate the natural gradient direction $s \approx H^{-1}g$, where $g$ is the policy gradient and $H$ is the FIM. A `fisher_vector_product` function avoids explicitly forming $H$.
6.  **Line Search:** After finding the step direction via conjugate gradient, a backtracking line search is performed to find the optimal step size that improves the surrogate objective while strictly adhering to the KL divergence constraint.

## Results
The `train_trpo_component.py` script was executed successfully.

- **Initial State:** The policy started with a total reward around -2.30.
- **Training Progression:** The model learned quickly, improving its reward progressively across epochs.
- **Final Performance:** The total reward consistently converged to 0.90 towards the later epochs.
- **Policy Evaluation:** The final evaluated policy effectively solved the simple environment, showing a high probability of taking the correct action to reach the terminal success state.
