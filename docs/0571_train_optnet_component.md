# Experiment: OptNet Component Training

**Script:** `train_optnet_component.py`
**Status:** Success

## Objective
Automatically generated report for the training and evaluation of the OptNet component.

## Methodology
The component was executed via the automated pipeline. The experiment implemented equality-constrained quadratic programming solving and derived gradients explicitly via KKT conditions to verify the backward pass.

## Results
```text
--- Testing OptNet Component ---
Forward Pass successful. Optimal z: [ 0.24581302  2.00625617 -2.20203512  3.32867073  0.80394323]
KKT Stationarity Residual norm: 2.504777312823609e-15
KKT Primal Feasibility Residual norm: 1.4392811567525336e-15

Analytical Gradients (dl/dp): [ -1.21195311  -7.523003     9.83628728 -15.04619589  -3.13211968]
Finite Diff Gradients  (dl/dp): [ -1.21195255  -7.5229892    9.83631084 -15.04614079  -3.13211693]
Gradient Error: 6.15511260e-05
OptNet backward pass verified successfully using Implicit Function Theorem.
```

## Conclusion
The component execution finished with status: Success. The backpropagation through the optimal solution using the implicit function theorem successfully matches numerical gradients.
