### Experiment 0568: Levenberg-Marquardt Component
**Script:** `train_levenberg_marquardt_component.py`
- **Objective**: Implement and verify the Levenberg-Marquardt algorithm mathematically for non-linear least squares optimization.
- **Methodology**: Implemented the algorithm in pure NumPy. Evaluated on a non-linear exponential model `y = a * exp(b * x)` using synthetic data with added Gaussian noise. The implementation computes the Jacobian, updates parameters via the damped Gauss-Newton method, and adapts the damping parameter (lambda) dynamically.
- **Outcome**: Successfully optimized parameters from an initial guess of [1.0, 1.0] to [2.4975, 1.3002], highly close to the true values [2.5, 1.3]. Final MSE was ~0.0084, validating robust convergence.
- **Conclusion**: The Levenberg-Marquardt component acts as a robust and efficient non-linear least squares solver and is verified for usage.
