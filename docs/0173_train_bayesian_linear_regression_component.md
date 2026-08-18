# Bayesian Linear Regression Component

## Overview
This experiment explores Bayesian Linear Regression mathematically and implements it from scratch using pure NumPy. Unlike standard linear regression, which provides point estimates for the weights, Bayesian Linear Regression provides a full posterior distribution over the weights. This allows the model to quantify its uncertainty about the predictions, which is crucial for active learning, reinforcement learning, and safety-critical applications.

## Mathematical Foundation
Bayesian Linear Regression places a prior over the weights $w$, typically a zero-mean Gaussian with covariance $\alpha^{-1} I$:
$$ p(w) = \mathcal{N}(w | 0, \alpha^{-1} I) $$

The likelihood of the data, assuming Gaussian noise with precision $\beta = 1/\sigma^2$, is:
$$ p(y | X, w) = \mathcal{N}(y | Xw, \beta^{-1} I) $$

By Bayes' theorem, the posterior over the weights is also Gaussian:
$$ p(w | X, y) = \mathcal{N}(w | m_N, S_N) $$
Where:
$$ S_N^{-1} = \alpha I + \beta X^T X $$
$$ m_N = \beta S_N X^T y $$

For a new input $x_{new}$, the predictive distribution is:
$$ p(y_{new} | x_{new}, X, y) = \mathcal{N}(y_{new} | m_N^T x_{new}, \sigma^2_N(x_{new})) $$
Where the predictive variance incorporates both the inherent noise and the model's uncertainty about the weights:
$$ \sigma^2_N(x_{new}) = \frac{1}{\beta} + x_{new}^T S_N x_{new} $$

## Implementation Details
- **Architecture**: Bayesian Linear Regression with conjugate Gaussian priors.
- **Inference**: Exact posterior updates via matrix inversion.
- **Key Parameters**: `alpha` (prior precision) and `beta` (noise precision).

## Results
The implementation was successfully tested on a synthetic dataset ($y = 2 + 3x + \text{noise}$). The model accurately inferred both the mean weights and the underlying noise level.

- **MSE**: ~0.2048
- **Learned Mean Weights**: Bias $\approx$ 1.9288 (True: 2.0), W $\approx$ 3.0280 (True: 3.0)
- **Average Predictive Std**: ~0.5049 (True Noise Std: 0.5)

## Conclusion
The Bayesian Linear Regression implementation successfully performs posterior inference, recovering the true underlying parameters while providing accurate predictive uncertainties. This foundation is essential for building more complex probabilistic models and for decision-making under uncertainty.
**Script:** `train_bayesian_linear_regression_component.py`
