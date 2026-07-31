# Experiment 0038: Conditional Flow Matching (CFM) Component

## Objective
Implement and verify a Continuous Normalizing Flow using Conditional Flow Matching (CFM) in pure NumPy. The goal is to mathematically model the straight-line probability flow ODE from a base Gaussian distribution to the data distribution, and train a neural network to predict the target vector field using manual backpropagation.

## Mathematical Formulation

### Forward Path
Let $x_0 \sim \mathcal{N}(0, I)$ be the base distribution and $x_1 \sim p_{data}$ be the data distribution.
The flow is defined as a straight path:
$x_t = (1 - t) x_0 + t x_1$
where $t \in [0, 1]$.

### Vector Field Objective
The target vector field (the derivative with respect to time $t$) is constant for a given pair:
$u_t(x_t|x_1) = x_1 - x_0$

The network $v_\theta(x_t, t)$ learns to approximate this vector field by minimizing the MSE loss:
$\mathcal{L} = \mathbb{E}_{t \sim U(0,1), x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$

## Results
- **Status:** Success
- **Final Loss:** 1.9855
- **Epochs:** 20000

## Conclusion
The model successfully learned to predict the target vector field mapping the base distribution to the data distribution, verifying the mathematical soundness of Conditional Flow Matching and its manual backpropagation.

**Script:** `train_flow_matching_component.py`
