# Experiment: Energy-Based Model (EBM) Training

**Script:** `train_ebm_component.py`
**Date:** 2024-08-04
**Status:** Success

## Description
Evaluated an Energy-Based Model (EBM) using pure NumPy. The script implements an EBM that assigns scalar energy values to input data, minimizing energy for true data and maximizing it for generated samples.

## Methodology
- **Architecture:** Two-layer MLP computing a scalar energy value.
- **Sampling:** Langevin Dynamics used to sample from the model distribution using gradients of the energy function with respect to the inputs.
- **Objective:** Contrastive Divergence approximating the gradient of the log-likelihood by comparing gradients from data samples and Langevin samples.

## Results
- The EBM successfully learned to assign lower energy to data points from the target distribution (a 2D ring) compared to random points.
- Langevin dynamics successfully produced samples closer to the target distribution by descending the energy landscape with added noise.
