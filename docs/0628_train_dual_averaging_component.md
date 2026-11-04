# Documentation for train_dual_averaging_component

# Experiment 0001: Dual Averaging Component
- **Objective**: Explore Dual Averaging mathematically for sparse online optimization.
- **Method**: Implemented a Dual Averaging optimizer in pure NumPy, testing its ability to solve linear regression by maintaining a dual average of past gradients and mapping it back to a primal parameter using a time-varying learning rate schedule.
- **Outcome**: The algorithm successfully minimized the mean squared error on synthetic data, validating the mechanism.
