# Logistic Regression Component

**Script:** `train_logistic_regression_component.py`

## Overview
Evaluates a Logistic Regression component mathematically in pure NumPy for binary classification.

## Mathematical Foundation
Logistic regression models the probability that a given input belongs to a particular class. It applies the sigmoid function to a linear combination of the input features:
$$ P(y=1|x) = \frac{1}{1 + e^{-(w^Tx + b)}} $$

The model parameters $w$ (weights) and $b$ (bias) are optimized using gradient descent to minimize the binary cross-entropy loss.

## Results
The implementation was successfully tested on a synthetic binary classification dataset, achieving a 100% accuracy, verifying the correctness of the forward pass and gradient updates.
