# Experiment: Lrp Component Training

**Script:** `train_lrp_component.py`
**Status:** Success

## Objective
Automatically generated report for the training and evaluation of the Lrp component.

## Methodology
The component was executed via the automated pipeline.

## Results
```text
Testing Layer-wise Relevance Propagation (LRP)...
Output: tensor([[-0.1914,  0.4712]], grad_fn=<AddmmBackward0>)
Input Relevance Scores: tensor([[-0.0580,  0.0516, -0.0500, -0.0400, -0.0354, -0.0801,  0.0383, -0.0181,
         -0.0505,  0.0166]], grad_fn=<MulBackward0>)
Sum of input relevance: -0.2257390171289444
Output of target class: -0.191369891166687
Conservation Error: 0.034369125962257385
LRP execution successful.
LRP component implemented and verified successfully.
```

## Conclusion
The component execution finished with status: Success.
