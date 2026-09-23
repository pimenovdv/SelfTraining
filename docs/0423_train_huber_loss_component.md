# Experiment: Huber Loss Component Training

**Script:** `train_huber_loss_component.py`
**Status:** Success

## Objective
Automatically generated report for the training and evaluation of the Huber Loss component.

## Methodology
The component was executed via the automated pipeline.

## Results
```text
Testing Huber Loss Component...
y_true: [1. 2. 3. 4. 5.]
y_pred: [ 1.1  2.   2.8 10.   4.5]
Delta: 1.0

Calculated Huber Loss: 1.1300
Calculated Gradient w.r.t y_pred:
[ 0.02 -0.   -0.04  0.2  -0.1 ]

Manual loss for index 0 (y_true=1.0, y_pred=1.1): 0.0050
Manual loss for index 3 (outlier, y_true=4.0, y_pred=10.0): 5.5000

Component test completed successfully.
```

## Conclusion
The component execution finished with status: Success.
