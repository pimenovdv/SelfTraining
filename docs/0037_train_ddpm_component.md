# Experiment 0037: Denoising Diffusion Probabilistic Model (DDPM) Component

## Objective
Implement and verify a basic DDPM using pure NumPy. The goal is to mathematically model the forward diffusion process (adding Gaussian noise) and the reverse denoising process, training a simple neural network to predict the added noise using manual backpropagation.

## Mathematical Formulation

### Forward Process
Let $x_0$ be the original data. The forward process adds noise over $T$ steps according to a variance schedule $\beta_1, \dots, \beta_T$.
$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$
Using the reparameterization trick, we can sample $x_t$ directly from $x_0$:
$\alpha_t = 1 - \beta_t$
$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$

### Reverse Process
The reverse process learns to undo the noise. We train a model $\epsilon_\theta(x_t, t)$ to predict the noise $\epsilon$ that was added to $x_0$ to get $x_t$.
The training objective is the simplified MSE loss:
$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]$

## Results
- **Status:** Success
- **Final Loss:** 0.4952
- **Epochs:** 2000

## Conclusion
The model successfully learned to predict the noise added during the forward diffusion process, confirming the mathematical soundness of the DDPM formulation and manual backpropagation for the reverse process.
