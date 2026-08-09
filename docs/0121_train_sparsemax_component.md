# Experiment 0121: Sparsemax Component

## Overview
This experiment verifies the implementation of the Sparsemax activation function. Sparsemax provides a differentiable alternative to Softmax that outputs exactly sparse probabilities, acting as a combination of Softmax and a sparsity-inducing regularization.

## Mathematical Basis
Sparsemax projects an input vector $z$ onto the probability simplex. Unlike Softmax, which is bounded strictly positive, the Euclidean projection in Sparsemax allows exact zero probabilities.

## Forward Pass
The projection involves sorting the input vector and finding a threshold $\tau(z)$ such that the sum of the positive shifted elements equals 1:
$p_i = \max(0, z_i - \tau(z))$

## Backward Pass
The gradient routes only through the elements that have non-zero probability (the support set):
$\\frac{\\partial L}{\\partial z} = S \odot (dp - \\frac{\\sum_{j \in S} dp_j}{|S|})$
where $S$ is the binary mask of the support set.

## Results
The model successfully converged.
Loss at end of training: 0.0085

Sample predictions demonstrated exact zeros, verifying the sparsity property of Sparsemax.
**Script:** `train_sparsemax_component.py`
