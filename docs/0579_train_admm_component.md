# ADMM (Alternating Direction Method of Multipliers) Component

**Script:** train_admm_component.py

## Overview
This document details the implementation and evaluation of the Alternating Direction Method of Multipliers (ADMM) algorithm, mathematically formulated for solving L1-regularized least squares (Lasso Regression). ADMM blends the decomposability of dual ascent with the superior convergence properties of the method of multipliers.

## Mathematical Formulation
ADMM solves optimization problems of the form:
$$ \min_x \frac{1}{2}\|Ax - b\|_2^2 + \alpha \|z\|_1 $$
subject to $x - z = 0$.

The augmented Lagrangian is:
$$ L_\rho(x, z, u) = \frac{1}{2}\|Ax - b\|_2^2 + \alpha \|z\|_1 + \frac{\rho}{2} \|x - z + u\|_2^2 $$

The ADMM updates are:
1. $x$-update: $x^{k+1} = (A^T A + \rho I)^{-1} (A^T b + \rho(z^k - u^k))$
2. $z$-update: $z^{k+1} = S_{\alpha/\rho}(x^{k+1} + u^k)$ (Soft Thresholding)
3. $u$-update: $u^{k+1} = u^k + x^{k+1} - z^{k+1}$

## Evaluation Metrics
- **Mean Squared Error:** 0.0067
- **Sparsity:** 14/50 zero coefficients
- **Execution Time:** 0.0097 seconds
- **Iterations:** 452

## Conclusion
The successful implementation demonstrates the convergence properties of ADMM, accurately enforcing sparsity while minimizing the reconstruction error.
