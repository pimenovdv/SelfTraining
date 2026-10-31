# Component Testing: Differential Evolution

**Script:** `train_differential_evolution_component.py`

## Objective
Evaluate the Differential Evolution method mathematically for non-convex, unconstrained global optimization.

## Methodology
Implemented Differential Evolution from scratch in pure NumPy on the 3-dimensional Rosenbrock function. Evaluated on the bounded region `[-5.0, 5.0]`. Used a population size of 50 and trained for up to 2000 iterations.

## Results
Differential Evolution successfully found the global minimum. The optimization converged to the parameters `[1.00001976 1.00003956 1.00007016]`, achieving a final loss of `0.000000`.

## Conclusion
Differential Evolution converges reliably for the non-convex Rosenbrock function, demonstrating the utility of evolutionary algorithms for multi-dimensional global optimization.
