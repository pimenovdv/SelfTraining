# BFGS Algorithm Component

## Objective
To implement and mathematically evaluate the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm, a highly popular quasi-Newton method for unconstrained optimization.

## Mathematical Formulation
BFGS finds stationary points by utilizing first-order gradient information to iteratively build an approximation of the inverse Hessian matrix.

The update rule for the approximated inverse Hessian $H$ is given by:
$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$
where:
$s_k = x_{k+1} - x_k$
$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$
$\rho_k = \frac{1}{y_k^T s_k}$

A backtracking line search ensuring the Armijo condition is utilized to find suitable step sizes.

## Implementation Details
**Script:** train_bfgs_component.py
- Pure NumPy implementation of the BFGS algorithm `bfgs_minimize`.
- Includes a `backtracking_line_search` helper function to enforce sufficient decrease.
- Evaluated on a convex 10-dimensional quadratic objective function.
- Evaluated on the highly non-convex multidimensional Rosenbrock function.

## Execution Results
- **Quadratic Function**: Converged precisely in 18 iterations, with an L2 error norm of 4.879500e-06 relative to the analytical solution. Execution took 0.0019 seconds.
- **Rosenbrock Function**: Efficiently navigated the non-convex valley, converging in 52 iterations to the optimal point [1.00000001, 1.0, 1.00000001, 1.00000002] with an optimal function value approaching 0.000000. Execution took 0.0084 seconds.

## Conclusion
The BFGS algorithm proved mathematically sound and computationally efficient. Its ability to accurately approximate the inverse Hessian utilizing only gradient information renders it highly scalable and effective for diverse unconstrained optimization tasks, correctly navigating both convex geometries and complex non-convex valleys.
