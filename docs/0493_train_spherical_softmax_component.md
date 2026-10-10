# Experiment: Spherical Softmax Component Training

**Script:** `train_spherical_softmax_component.py`
**Status:** Success

## Objective
Automatically generated report for the training and evaluation of the Spherical Softmax component.

## Methodology
The component was executed via the automated pipeline.

## Results
```text
Testing Spherical Softmax Component...
Input:
tensor([[ 1.,  2.,  3.],
        [-1.,  0.,  1.]])
Output:
tensor([[0.0714, 0.2857, 0.6429],
        [0.5000, 0.0000, 0.5000]])
Sum along dim=-1:
tensor([1., 1.])
Spherical Softmax tests passed.
Starting training...
Epoch 20, Loss: 1.3541
Epoch 40, Loss: 1.3264
Epoch 60, Loss: 1.3101
Epoch 80, Loss: 1.2967
Epoch 100, Loss: 1.2766
Training finished.
```

## Conclusion
The component execution finished with status: Success.
