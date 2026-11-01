# Cross-Entropy Method Component

**Script:** train_cem_component.py

## Overview
This experiment implements the Cross-Entropy Method (CEM) for continuous optimization problems, tested specifically on the Rosenbrock function. CEM is a derivative-free optimization method that frames the problem as estimating the probability of rare events. It maintains a parameterized distribution (in this case, Gaussian) over the solution space and updates its parameters (mean and standard deviation) iteratively based on the best-performing ("elite") samples.

## Implementation Details
- Uses a Gaussian distribution parameterized by mean and standard deviation.
- Evaluates a population of candidate solutions at each step.
- Selects the top 10% (elite fraction) of solutions to recalculate the mean and standard deviation for the next generation's distribution.
- Includes a small constant to standard deviation updates for numerical stability.

## Results
- Final Best Eval: 0.050295
- The algorithm successfully converged towards the global minimum (1, 1).
- Training complete and successful.
