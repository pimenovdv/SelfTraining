# Experiment: Irl Component Training

**Script:** `train_irl_component.py`
**Status:** Success

## Objective
Automatically generated report for the training and evaluation of the Irl component.

## Methodology
The component was executed via the automated pipeline.

## Results
```text
Testing Inverse Reinforcement Learning (IRL) Component...
Training MaxEnt IRL proxy for 200 epochs...
Epoch   0 | Gradient Norm: 0.1497 | Weights: [0.36553793 0.87014882 0.3277764 ]
Epoch  20 | Gradient Norm: 0.0493 | Weights: [0.65487544 0.05418662 0.41002901]
Epoch  40 | Gradient Norm: 0.0211 | Weights: [ 0.71462503 -0.21508019  0.44159325]
Epoch  60 | Gradient Norm: 0.0123 | Weights: [ 0.71348579 -0.31689185  0.44812564]
Epoch  80 | Gradient Norm: 0.0096 | Weights: [ 0.69914729 -0.36034787  0.44747844]
Epoch 100 | Gradient Norm: 0.0089 | Weights: [ 0.68492295 -0.38135303  0.44556642]
Epoch 120 | Gradient Norm: 0.0087 | Weights: [ 0.67379676 -0.39273483  0.44390227]
Epoch 140 | Gradient Norm: 0.0087 | Weights: [ 0.66578369 -0.39946612  0.44270873]
Epoch 160 | Gradient Norm: 0.0088 | Weights: [ 0.66021646 -0.40368073  0.4419038 ]
Epoch 180 | Gradient Norm: 0.0088 | Weights: [ 0.65641421 -0.40640854  0.44137148]
Epoch 199 | Gradient Norm: 0.0088 | Weights: [ 0.65394556 -0.40813251  0.44103503]

--- Final Results ---
True Weights: [ 0.8 -0.4  0.5]
Recovered Weights: [ 0.65394556 -0.40813251  0.44103503]
Success! The IRL component successfully minimized the gradient norm, dynamically aligning feature expectations via Value Iteration.
```

## Conclusion
The component execution finished with status: Success.
