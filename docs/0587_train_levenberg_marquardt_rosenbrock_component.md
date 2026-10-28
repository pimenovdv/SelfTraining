# Experiment: Levenberg-Marquardt Optimization Component on Rosenbrock Function

## Objective
Implement and verify the Levenberg-Marquardt (LM) algorithm mathematically in pure NumPy, testing its ability to solve non-linear least squares problems by interpolating between the Gauss-Newton algorithm and the method of gradient descent.

## Method
- **Algorithm:** The Levenberg-Marquardt algorithm iteratively updates parameters using the normal equations augmented with a damping parameter $\lambda$.
  - Update rule: $(J^T J + \lambda \text{diag}(J^T J)) \delta = -J^T r$
  - The damping parameter $\lambda$ is adjusted based on the gain ratio $\rho$, which compares actual reduction in the cost function to the reduction predicted by a linear model.
- **Problem Setup:** The algorithm is tested on the Rosenbrock function, a non-convex function commonly used as a performance test problem for optimization algorithms.
- **Implementation:** The optimization logic is written in pure NumPy. The `rosenbrock_residual` and `rosenbrock_jacobian` functions are defined to compute the residuals and their Jacobian matrix.
- **Script:** `train_levenberg_marquardt_rosenbrock_component.py`

## Results
- The algorithm successfully converged to the true minimum of the Rosenbrock function `[1.0, 1.0]`.
- The implementation effectively adjusts the damping parameter to ensure robust convergence, avoiding singularities in the Jacobian and handling the non-convexity of the Rosenbrock function.

## Conclusion
The pure NumPy implementation of the Levenberg-Marquardt algorithm functions correctly, demonstrating its capacity for efficient non-linear least squares optimization by dynamically balancing between Gauss-Newton and gradient descent steps.
