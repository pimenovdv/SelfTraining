# COBYLA Optimization

**Script:** train_cobyla_component.py

## Overview
This experiment explores Constrained Optimization BY Linear Approximations (COBYLA), a derivative-free optimization method for constrained problems. COBYLA works by iteratively constructing linear approximations of the objective function and the constraints.

## Methodology
The `train_cobyla_component.py` script implements the COBYLA algorithm to minimize a simple quadratic objective function $f(x, y) = x^2 + y^2$ subject to the linear inequality constraint $x + y \geq 1$. This is solved using `scipy.optimize.minimize` with `method='COBYLA'`. The initial point was set to `(2.0, 2.0)`.

## Results
- **Execution Time:** ~0.09 seconds
- **Final Minimum:** 0.500000
- **Optimal Parameters:** `[0.50000742, 0.49999258]`
- **Function Evaluations:** 37

## Conclusion
The COBYLA algorithm successfully converged to the constrained minimum, satisfying the constraint $x+y=1$. It provides a reliable method for constrained optimization where derivatives are either unavailable or difficult to compute.
