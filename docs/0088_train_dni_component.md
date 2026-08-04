# Experiment: Decoupled Neural Interfaces (DNI)

**Script:** `train_dni_component.py`
**Date:** 2024-08-04
**Status:** Success

## Description
Evaluated Decoupled Neural Interfaces (DNI) using Synthetic Gradients with pure NumPy. The script implements DNI to allow layers to be updated asynchronously, breaking the standard backpropagation forward-backward lock.

## Methodology
- **Architecture:** A 3-layer MLP where the first two hidden layers use Synthetic Gradients.
- **Mechanism:** Each hidden layer contains an auxiliary neural network (a linear layer here) that predicts its own error gradient based on its activation.
- **Optimization:** The primary weights are updated using the *synthetic* gradient. The auxiliary network is trained by comparing its synthetic gradient prediction against the *true* gradient that eventually flows back.

## Results
- The network successfully converged on the XOR-like dataset, achieving high accuracy.
- This verifies that layers can effectively learn from local gradient approximations without waiting for the full network backward pass.
