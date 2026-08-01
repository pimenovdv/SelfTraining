# Experiment: Reversible Residual Networks (RevNet)

## Objective
To mathematically model and implement a Reversible Residual Network (RevNet) block from scratch using `numpy`, verifying its ability to train without storing intermediate activations for backpropagation, thereby saving memory.

## Mathematical Formulation

A RevNet block splits the input $x$ into two halves, $x_1$ and $x_2$.
The forward pass is defined as:
$$y_1 = x_1 + F(x_2)$$
$$y_2 = x_2 + G(y_1)$$

where $F$ and $G$ are arbitrary residual functions (e.g., MLPs or CNNs).

The key feature of RevNets is that the input $x$ can be exactly reconstructed from the output $y$ during the backward pass:
$$x_2 = y_2 - G(y_1)$$
$$x_1 = y_1 - F(x_2)$$

This allows computing gradients during backpropagation without storing the intermediate activations (except for the current block being computed), significantly reducing the memory footprint from $O(L)$ to $O(1)$ for storing activations, where $L$ is the number of layers.

Gradients are computed as:
$$\frac{\partial L}{\partial y_1} = \frac{\partial L}{\partial y_1} + \frac{\partial L}{\partial y_2} \frac{\partial G(y_1)}{\partial y_1}$$
$$\frac{\partial L}{\partial x_2} = \frac{\partial L}{\partial y_2} + \frac{\partial L}{\partial y_1} \frac{\partial F(x_2)}{\partial x_2}$$
$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial y_1}$$

## Implementation Details
- Created a `SimpleMLP` class to act as the residual functions $F$ and $G$.
- Implemented the `RevNetBlock` class with `forward` and `backward` methods.
- The `backward` method first reconstructs the inputs $x_1, x_2$ from $y_1, y_2$ and then computes the gradients for $F$ and $G$.
- Integrated the RevNet block into a training loop with a final linear layer, optimizing an MSE loss.

## Results
The `numpy` implementation successfully trained on synthetic data. The loss decreased steadily over the epochs, demonstrating that the exact input reconstruction and gradient computation via the reversible equations are mathematically sound and implementable.

## Conclusion
RevNets offer a powerful architectural paradigm for memory-efficient training of deep networks. This experiment validates the core reversible mechanism, which can be extended to deeper networks and more complex residual functions.


**Script:** `train_revnet_component.py`
