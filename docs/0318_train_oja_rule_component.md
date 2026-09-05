# Experiment: Oja Rule Component Training

**Script:** `train_oja_rule_component.py`
**Status:** Success

## Objective
Automatically generated report for the training and evaluation of the Oja Rule component.

## Methodology
The component was executed via the automated pipeline.

## Results
```text
Starting Oja's Rule (Hebbian Learning) Component Training...
Dataset generated. Input dimension: 3, Output dimension (PCs): 2
True Top 2 Principal Components:
[[-0.8906577  -0.44437652 -0.09622044]
 [-0.4381531   0.89538382 -0.07943341]]
Epoch [5/20] completed.
Epoch [10/20] completed.
Epoch [15/20] completed.
Epoch [20/20] completed.
Training completed in 0.46 seconds.

Learned Weights (normalized):
[[ 0.89624405  0.43196782  0.10074951]
 [-0.89624405 -0.43196785 -0.10074951]]

Orthogonality check (W * W^T):
[[ 1.0000001 -1.0000001]
 [-1.0000001  1.0000001]]
Model saved to results/oja_rule_model.pt
```

## Conclusion
The component execution finished with status: Success.
