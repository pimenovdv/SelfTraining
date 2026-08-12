# Experiment: Scaled Exponential Linear Unit (SELU)

**Status:** Success
**Script:** `train_selu_component.py`

## Objective
To mathematically implement and evaluate a Scaled Exponential Linear Unit (SELU) component in pure NumPy. SELU introduces self-normalizing properties to deep neural networks, maintaining a mean of 0 and variance of 1 across layers without requiring explicit normalization layers like Batch Normalization.

## Mathematical Foundation
The SELU activation function is defined as:
`SELU(x) = \lambda x` if `x > 0`
`SELU(x) = \lambda \alpha (\exp(x) - 1)` if `x \le 0`

Where `\alpha` and `\lambda` are constants derived mathematically to induce self-normalization:
*   `\alpha \approx 1.67326`
*   `\lambda \approx 1.0507`

The derivative is:
`d(SELU(x)) / dx = \lambda` if `x > 0`
`d(SELU(x)) / dx = \lambda \alpha \exp(x)` if `x \le 0`

## Results
The component successfully trains a deep non-linear regression model using SELU activations. The initial checks confirm that the variance of activations across layers remains close to 1, demonstrating the self-normalizing property of SELU. The loss decreases steadily during training.

## Conclusion
The pure NumPy implementation of the SELU activation and its derivative functions correctly, enabling successful learning and self-normalization in deep networks.
