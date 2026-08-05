# Experiment 0095: Conditional Neural Process (CNP)

**Objective:** Implement and verify a Conditional Neural Process (CNP) mathematically.

**Methodology:** The CNP learns to model distributions over functions (meta-learning) by processing context points $(x_c, y_c)$ through an encoder to form a fixed-size representation, aggregating this representation, and decoding it along with target inputs $x_t$ to predict the mean and variance of $y_t$. Trained on a family of sine waves using manual backpropagation with Negative Log-Likelihood.

**Results:**
- Initial Loss: 281.1806
- Final Loss: -0.4231
- Success: True

**Conclusion:** The CNP component successfully learned to condition on varying context points to predict the distribution of target points for an entire family of sine wave functions, demonstrating few-shot function approximation.
**Script:** `train_cnp_component.py`
