# Experiment 0116: Sharpness-Aware Minimization (SAM)

**Script:** `train_sam_component.py`

## Objective
Implement and verify Sharpness-Aware Minimization (SAM) using pure NumPy. The goal is to mathematically model the process of simultaneously minimizing the loss value and loss sharpness by finding a parameter perturbation that maximizes the loss, and then computing the gradient at this perturbed parameter to update the model.

## Mathematical Formulation
Let $\rho > 0$ be the neighborhood size. For a batch of data, SAM approximates the solution to a min-max optimization problem:
$\min_w \max_{||\epsilon||_2 \leq \rho} L(w + \epsilon)$

To approximate the inner maximization, SAM computes a first-order Taylor expansion:
$\epsilon^*(w) \approx \arg\max_{||\epsilon||_2 \leq \rho} \epsilon^T \nabla_w L(w) = \rho \frac{\nabla_w L(w)}{||\nabla_w L(w)||_2}$

Then, the final gradient used to update the weights is computed at the perturbed weights:
$w_{t+1} = w_t - \eta \nabla_w L(w_t + \epsilon^*(w_t))$

## Results
- **Status:** Success
- **Final Loss:** 0.0224

## Conclusion
The MLP was successfully trained using SAM, minimizing the loss over a synthetic non-linear dataset while incorporating the sharpness penalty through weight perturbations during the forward pass.
