# Experiment: Softmax Component Training

**Script:** `train_softmax_component.py`
**Status:** Success

## Objective
Automatically generated report for the training and evaluation of the Softmax component.

## Methodology
The component was executed via the automated pipeline.

## Results
```text
Testing Softmax Component
-------------------------
Input x:
[[   1.    2.    3.]
 [  -1.   -2.   -3.]
 [1000. 1000. 1000.]]

Softmax output:
[[0.09003057 0.24472847 0.66524096]
 [0.66524096 0.24472847 0.09003057]
 [0.33333333 0.33333333 0.33333333]]

Sums of probabilities (should be 1s): [1. 1. 1.]

Simulated grad_output:
[[ 0.1  0.2 -0.3]
 [-0.1  0.1  0. ]
 [ 0.   0.   0. ]]

Gradient input (backprop):
[[ 0.02175351  0.08360501 -0.10535851]
 [-0.03854988  0.03476398  0.0037859 ]
 [ 0.          0.          0.        ]]

Softmax component successfully passed all tests.
```

## Conclusion
The component execution finished with status: Success.
