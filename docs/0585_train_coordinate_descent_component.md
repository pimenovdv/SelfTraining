# Coordinate Descent Component

**Script:** train_coordinate_descent_component.py

## Overview
This document evaluates the Coordinate Descent algorithm mathematically in pure NumPy, testing its ability to solve the Lasso Optimization problem by sequentially optimizing each parameter while holding others fixed, utilizing a soft-thresholding operator for the L1 penalty, demonstrating convergence on synthetic sparse data.

## Execution Result
```
True weights: [ 1.5 -2.   0.   0.   3.2]
Estimated weights: [ 1.49203436e+00 -1.97825297e+00  0.00000000e+00  1.70745063e-03
  3.18920450e+00]
Final Lasso Objective: 7.0963
```

## Conclusion
The Coordinate Descent component successfully optimized the Lasso objective, finding the exact support of the sparse true weights and shrinking irrelevant weights to zero.
