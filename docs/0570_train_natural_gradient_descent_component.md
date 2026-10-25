# Experiment: Natural Gradient Descent Component Training

**Script:** `train_natural_gradient_descent_component.py`
**Status:** Success

## Objective
Automatically generated report for the training and evaluation of the Natural Gradient Descent component.

## Methodology
The component was executed via the automated pipeline.

## Results
```text
Initializing Natural Gradient Descent Component...
Data shape: (200, 4), Targets shape: (200,)
True weights: [ 1.5  -2.    0.5   3.14]

Training with Natural Gradient Descent...

Final Learned Weights (NGD): [ 1.52703777 -1.98864084  0.50357583  3.12031589]
Final Loss (NGD): 0.1322

Training with Standard Gradient Descent for comparison...
Final Learned Weights (Standard GD): [ 0.71129537 -2.00993146  0.64580535  3.14546818]
Final Loss (Standard GD): 0.3876

Observation: NGD converges much faster and is invariant to feature scaling,
while standard GD struggles with the ill-conditioned input matrix.
Component implemented successfully.
```

## Conclusion
The component execution finished with status: Success.
