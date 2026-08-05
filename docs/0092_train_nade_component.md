# Experiment 0092: Neural Autoregressive Distribution Estimator (NADE)

**Objective:** Implement and verify a Neural Autoregressive Distribution Estimator (NADE) to model joint probability distributions of binary data.

**Methodology:** NADE factors the joint distribution into a product of conditional distributions. We train it to minimize the Binary Cross Entropy (Negative Log-Likelihood) on a synthetic sequential binary dataset.

**Results:**
- Initial Loss: 3.3528
- Final Loss: 1.9435
- Success: True

**Conclusion:** The NADE component successfully learned the conditional probabilities of the binary dataset, confirming its capability for exact likelihood estimation and autoregressive generation.
**Script:** `train_nade_component.py`
