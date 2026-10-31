# Experiment: Firefly Algorithm Component Training

**Script:** `train_firefly_algorithm_component.py`
**Status:** Success

## Objective
Implement and evaluate the Firefly Algorithm mathematically in pure NumPy, testing its ability to solve non-convex global optimization problems inspired by the flashing behavior of fireflies.

## Methodology
- **Algorithm:** The Firefly Algorithm is a metaheuristic, nature-inspired, optimization algorithm which relies on the assumptions that all fireflies are unisex and are attracted to each other.
- **Attraction Mechanism:** The attractiveness of a firefly is proportional to its brightness (objective function value), and it decreases with distance. Less bright fireflies will move towards brighter ones. If there is no brighter firefly, it moves randomly.
- **Objective Function:** We optimized the non-convex Rosenbrock function in 3 dimensions, evaluating the algorithm's global search capability.
- **Implementation:** Built mathematically in pure NumPy, explicitly updating population vectors based on relative attractiveness, exponential light absorption, and a randomization parameter $\alpha$.
- **Script:** `train_firefly_algorithm_component.py`

## Results
```text
Firefly Algorithm Component Test
Final Loss: 0.000000
Execution Time: 9.0382 seconds
```

## Conclusion
The component execution finished with status: Success. The pure NumPy implementation successfully optimized the Rosenbrock function, demonstrating its capacity for effective derivative-free global optimization by balancing local exploitation (attraction) and global exploration (random walk).
