# Experiment: Gaussian Process Regression (GPR) Component

**Goal:** Explore probabilistic non-parametric modeling mathematically by implementing a Gaussian Process Regression (GPR) component in pure NumPy.

**Script:** `train_gaussian_process_component.py`

**Hypothesis:** By employing a Gaussian process with an RBF kernel, we can perform non-parametric regression on training data and provide mathematically sound predictive uncertainty (variance) for unobserved points, allowing for robust interpolation of nonlinear functions.

**Description:**
This experiment implements Gaussian Process Regression from scratch. The code initializes training data (a noisy sine wave) and uses a Radial Basis Function (RBF) kernel to compute covariance matrices. It employs Cholesky decomposition for numerical stability when computing the predictive mean and variance for test points. We evaluate the model based on its training MSE and its ability to provide predictive standard deviations (uncertainty bounds), as well as computing the log marginal likelihood.

**Results:**
- **Execution:** Successful.
- **Log Marginal Likelihood:** ~ -5.9292
- **Training MSE:** ~ 0.0046
- **Mean Test Std (Uncertainty):** ~ 0.2420

**Conclusions:**
The GPR component successfully modeled the non-linear dataset while natively producing principled uncertainty estimates. The Cholesky decomposition correctly handles the matrix inversion robustly. The test highlights the utility of Gaussian Processes for providing confidence intervals alongside predictions, which is critical for active learning, Bayesian optimization, and safety in downstream AGI components.
