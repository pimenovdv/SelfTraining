# Neural Radiance Field (NeRF) Component Training

**Script:** `train_nerf_component.py`
**Date:** 2026-08-11
**Status:** Success

## Mathematical Background

Neural Radiance Fields (NeRF) represent a continuous 3D scene as a function mapped by a neural network. The input is a 3D coordinate (often with viewing direction, omitted here for simplicity), and the output is volume density $\sigma$ and color $c = (r, g, b)$.

Volume rendering along a ray is computed using the integral:
$$ C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) c(\mathbf{r}(t), \mathbf{d}) dt $$

Where $T(t)$ is the accumulated transmittance:
$$ T(t) = \exp\left(-\int_{t_n}^t \sigma(\mathbf{r}(s)) ds\right) $$

This script implements the discrete approximation using classical alpha compositing, and computes full gradients mathematically in pure NumPy.

## Experiment Details

- Modeled a single ray with 64 sample points.
- Network: MLP with 2 hidden layers (64 units), Softplus activation for density, Sigmoid for color.
- Target Color: [0.9 0.1 0.5]
- Final Rendered Color: [0.9 0.1 0.5]
- Final Loss: 0.000000
