# Experiment: FISTA Component Training

**Script:** `train_fista_component.py`
**Status:** Success

## Objective
Implement and verify the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) mathematically for solving L1-regularized least squares (Lasso optimization).

## Methodology
Implemented FISTA in pure NumPy. The algorithm uses a proximal gradient method, alternating between a standard gradient descent step on the smooth least-squares objective and a soft-thresholding proximal operator for the non-smooth L1 penalty, accelerated by Nesterov momentum. Tested on synthetic data with a known sparse ground truth.

## Results
```text
Testing FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) Component...
Mean Squared Error: 0.0235
Sparsity: 45/50 weights are exactly zero
Estimated weights (first 10): [ 4.94045176 -2.9406871   1.94949475 -0.95075003  3.95304115 -0.
 -0.         -0.         -0.         -0.        ]
True weights (first 10): [ 5. -3.  2. -1.  4.  0.  0.  0.  0.  0.]
Execution time: 0.0205 seconds
FISTA component successfully optimized the L1-regularized objective.
```

## Conclusion
The component execution finished with status: Success. FISTA successfully found a sparse solution and efficiently optimized the non-smooth objective, converging to the true sparse signal.
