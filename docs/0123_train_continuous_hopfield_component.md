# Experiment 0123: Continuous Hopfield Network

## Overview
This experiment implements the Continuous (Modern) Hopfield Network mathematically in pure NumPy. Continuous Hopfield Networks generalize classic binary Hopfield networks to continuous states and use an exponential interaction function (log-sum-exp energy), massively increasing storage capacity.

## Mathematical Basis
The energy function of the Continuous Hopfield Network is given by:
$E(\xi) = -\frac{1}{\beta} \log \sum_{i=1}^N \exp(\beta x_i^T \xi) + \frac{1}{2} \xi^T \xi$

Where $X = (x_1, \dots, x_N)$ are the stored patterns, $\xi$ is the state vector, and $\beta$ is the inverse temperature parameter.
The update rule that minimizes this energy is:
$\xi^{new} = X^T \text{softmax}(\beta X \xi)$

This update rule is mathematically equivalent to the Self-Attention mechanism used in Transformer architectures, bridging associative memory and attention.

## Results
The implementation successfully retrieves stored continuous patterns from noisy initializations, iteratively minimizing the continuous energy function.
**Script:** `train_continuous_hopfield_component.py`
