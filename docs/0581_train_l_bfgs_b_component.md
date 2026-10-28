# Component: L-BFGS-B Optimization

## Description
This component implements the **L-BFGS-B** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds) algorithm for mathematical optimization. L-BFGS-B is a quasi-Newton method that approximates the inverse Hessian matrix to guide the optimization process. It is highly efficient for problems with a large number of variables as it maintains only a limited history of gradients and parameter updates, minimizing memory overhead. Furthermore, it explicitly handles simple bound constraints on the optimization variables by identifying active sets at each iteration.

The component is tested against the classic non-convex **Rosenbrock function**, using analytically computed gradients, to demonstrate its capability in navigating complex loss landscapes while strictly adhering to imposed variable bounds.

**Script:** train_l_bfgs_b_component.py

## Mathematical Context
The L-BFGS-B algorithm addresses the minimization problem:
$$ \min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) $$
subject to bound constraints:
$$ l_i \leq x_i \leq u_i, \quad i = 1, \dots, n $$

Instead of fully computing and inverting the $n \times n$ Hessian matrix $\nabla^2 f(\mathbf{x})$, L-BFGS-B approximates the action of the inverse Hessian on a vector using the last $m$ changes in positions ($\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$) and gradients ($\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$).

To handle the bound constraints, at each iteration $k$, the algorithm uses the gradient and the current bounded parameters to define a "Cauchy point". Variables that are at their bounds and where the gradient pushes them further outside the bounds are classified as the "active set" and held fixed. The remaining "free variables" are updated using an unconstrained L-BFGS step within the permissible subspace.

## Execution and Results
The algorithm was initialized at $\mathbf{x}_0 = [-1.2, 1.0]$ with bounds set to $[-2.0, 2.0]$ for both variables.

**Results:**
- Optimal parameters found: `[0.99999999, 0.99999997]` (close to the true global minimum of `[1.0, 1.0]`)
- Final Loss: `0.000000`
- Execution Time: ~`0.0038` seconds

## Conclusion
The L-BFGS-B implementation successfully and rapidly optimized the non-convex Rosenbrock function while respecting the established boundaries. This component serves as a robust foundation for larger-scale parametric optimization problems within the AGI sandbox, especially where memory limitations or explicit parameter ranges are critical considerations.
