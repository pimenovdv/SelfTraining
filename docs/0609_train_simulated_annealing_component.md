# Simulated Annealing Component

**Script:** train_simulated_annealing_component.py

## Overview
This experiment implements Simulated Annealing for continuous optimization problems, specifically testing it on the Rosenbrock function. Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function, inspired by the annealing process in metallurgy.

## Implementation Details
- Uses random perturbations scaled by a dynamic temperature parameter.
- Temperature decreases geometrically (cooling schedule).
- Accepts worse solutions with a probability dependent on the temperature and energy difference, helping to escape local minima.
- Tested on a 2D Rosenbrock function.

## Results
- Final Best Eval: 0.000162
- Training complete and successful.
- The algorithm successfully converged close to the global minimum (1, 1).
