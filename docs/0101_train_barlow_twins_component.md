# Experiment 0101: Barlow Twins

**Script:** `train_barlow_twins_component.py`

## Objective
Evaluate a Barlow Twins component for non-contrastive self-supervised learning, verifying its ability to prevent representation collapse by driving the cross-correlation matrix between representations of distorted versions of a sample to the identity matrix.

## Configuration
- Input Dimension: 16
- Hidden Dimension: 64
- Projection Dimension: 16
- Batch Size: 128
- Epochs: 200
- Learning Rate: 0.05
- Lambda (Off-diagonal weight): 0.005

## Results
- Initial Loss: 0.1076
- Final Loss: 0.0474
- Training Time: 2.82s
- Success: True

## Conclusion
The model successfully minimize the Barlow Twins loss, driving the cross-correlation matrix toward the identity matrix, confirming that representations were learned without collapsing into trivial constant solutions.
