# Proximal Gradient Descent Component

**Script:** train_proximal_gradient_descent_component.py

## Overview
This document details the implementation and evaluation of Proximal Gradient Descent, mathematically formulated for optimizing non-differentiable objective functions like L1-regularized least squares (Lasso Regression). It combines standard gradient descent on the smooth part with a proximal operator (soft thresholding) for the non-smooth penalty.

## Mathematical Formulation
Proximal Gradient Descent solves optimization problems of the form:
$$ \min_w g(w) + h(w) $$
where $g(w)$ is smooth and differentiable, and $h(w)$ is convex but possibly non-differentiable.

For Lasso regression:
- $g(w) = \frac{1}{2m} \|Xw - y\|_2^2$ (Mean Squared Error)
- $h(w) = \lambda \|w\|_1$ (L1 Regularization)

The update rule at iteration $k$ is:
1. **Gradient Step (on smooth part):** $w_{temp} = w_k - \eta \nabla g(w_k)$
2. **Proximal Step (on non-smooth part):** $w_{k+1} = \text{prox}_{\eta h}(w_{temp})$

For $h(w) = \lambda \|w\|_1$, the proximal operator is the Soft Thresholding function:
$$ S_{\eta \lambda}(x) = \text{sign}(x) \max(|x| - \eta \lambda, 0) $$

## Evaluation Metrics
- **Mean Squared Error:** 1.2628
- **Sparsity:** 46/50 zero coefficients
- **Execution Time:** 0.0079 seconds

## Conclusion
The successful implementation demonstrates how Proximal Gradient Descent effectively enforces sparsity in linear models via the soft-thresholding operator while minimizing the loss, bridging the gap between smooth optimization and non-differentiable penalties.
