# SVGD Component

**Script:** `train_svgd_component.py`

**Objective:** To implement and mathematically verify Stein Variational Gradient Descent (SVGD), a deterministic sampling algorithm that uses interacting particles to approximate a target distribution.

**Method:** Implemented SVGD where a set of particles are initialized randomly and iteratively updated to match a 1D target Gaussian distribution ($\mathcal{N}(\mu=2, \sigma=0.5)$). The update direction is determined by an RBF kernel which repels particles to ensure diverse exploration, driven by the score function (gradient of the log PDF) of the target distribution.

**Results:** The SVGD implementation successfully approximated the target Gaussian distribution. With 100 particles, the estimated mean converged to 1.9999 (target 2.0) and the standard deviation converged to 0.4908 (target 0.5), perfectly capturing the underlying distribution.

**Next Steps:** This component is useful for Bayesian inference tasks as a deterministic alternative to MCMC, efficiently approximating complex posteriors.
