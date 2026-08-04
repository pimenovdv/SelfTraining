# Experiment: Evolution Strategies (ES)

**Script:** `train_es_component.py`
**Date:** 2024-08-04
**Status:** Success

## Description
Evaluated an Evolution Strategies (ES) component using pure NumPy. The script implements gradient-free optimization of a neural network by perturbing parameters, evaluating fitness, and applying updates based on the population's performance.

## Methodology
- **Architecture:** Two-layer MLP.
- **Task:** Non-linear regression using sine and cosine functions.
- **Optimization:** Evolution Strategies via random noise injection, evaluating symmetric perturbations (+ and -), and estimating the gradient of expected fitness.

## Results
- The network successfully minimized the Mean Squared Error (maximized negative MSE fitness) without using backpropagation.
- Initial fitness was approximately -2.3, improving to approximately -0.01 after 400 epochs.
