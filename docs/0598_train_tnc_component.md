# Experiment: train_tnc_component

**Script:** train_tnc_component.py
**Date:** 2023-10-30

## Objective
Evaluate the Truncated Newton (TNC) optimization method mathematically for non-convex, unconstrained optimization.

## Methodology
Implemented TNC using `scipy.optimize.minimize` on the 5-dimensional Rosenbrock function, providing the exact Jacobian (`rosenbrock_der`). The start point was the origin `[0,0,0,0,0]`.

## Results
- **Success:** True
- **Time:** ~0.0063 seconds
- **Message:** Converged (|f_n-f_(n-1)| ~= 0)
- **Iterations:** 22
- **Final Parameters:** `[1.00000027, 1.00000083, 1.00000189, 1.00000388, 1.00000775]`
- **Final Loss (f(x)):** 3.4175507503021923e-11

## Conclusion
TNC converges reliably for the non-convex Rosenbrock function, acting efficiently as a Newton conjugate-gradient method for unconstrained optimization. It is highly suitable for large-scale problems with bounds constraints.
