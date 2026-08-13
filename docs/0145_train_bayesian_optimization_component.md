# Experiment Report: Bayesian Optimization Component

## Overview
This experiment implements and tests Bayesian Optimization mathematically using pure NumPy. Bayesian Optimization is a sequential design strategy for global optimization of black-box functions. It builds a probabilistic surrogate model (Gaussian Process) of the objective function and uses an acquisition function (Expected Improvement) to decide where to evaluate next, balancing exploration and exploitation.

**Script:** `train_bayesian_optimization_component.py`
**Description:** Evaluates a Bayesian Optimization component mathematically in pure NumPy, testing its ability to find the global maximum of a non-convex 1D function efficiently by fitting a Gaussian Process and maximizing the Expected Improvement acquisition function.

## Results
- **Success:** Yes.
- The Bayesian Optimization efficiently found a near-optimal maximum of the complex non-convex function $f(x) = x \sin(x)$ within [0, 10].
- The model correctly balanced exploration and exploitation via the Gaussian Process predictive uncertainty and the Expected Improvement calculation.

## Mathematical Formulation
The algorithm utilizes a Gaussian Process (GP) to model the objective function:
$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$
where $m(x)=0$ and $k(x, x')$ is the RBF kernel.

The Expected Improvement (EI) acquisition function is defined as:
$EI(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)]$
$EI(x) = (\mu(x) - f(x^+) - \xi)\Phi(Z) + \sigma(x)\phi(Z)$
where $Z = \frac{\mu(x) - f(x^+) - \xi}{\sigma(x)}$, $\Phi$ is the CDF of the standard normal distribution, and $\phi$ is the PDF of the standard normal distribution.

## Next Steps
- Integrate Bayesian Optimization for hyperparameter tuning of other components within the repository.
- Explore other acquisition functions such as Upper Confidence Bound (UCB).
