# Experiment 0125: Adaptive Computation Time (ACT) Component

## Objective
To implement and verify Adaptive Computation Time (ACT), enabling a neural network to dynamically determine its own computation depth (number of processing steps) per input, minimizing a ponder cost alongside the task loss.

## Description
This experiment tests a pure NumPy implementation of the ACT mechanism. For each input, the network iteratively updates its hidden state and computes a halting probability. The final state is a weighted average of intermediate states, with weights determined by the halting probabilities. A ponder penalty encourages halting earlier.

**Script:** `train_act_component.py`

## Hypothesis
By introducing a ponder cost and a differentiable halting mechanism, the network can learn to use fewer computation steps when possible, while retaining the capacity to process inputs more deeply if required by the task, fully supported by exact manual gradients.

## Results
- The ACT component successfully minimized the task loss.
- The average ponder steps converged to a stable value, demonstrating the balance between task performance and ponder cost.
- Manual backpropagation successfully routed gradients through the dynamic computation graph, including the ponder probabilities and weights.

## Conclusion
The ACT component provides a mathematically sound foundation for dynamically scaling compute at inference and training time, which is essential for efficient, scalable AGI models.
