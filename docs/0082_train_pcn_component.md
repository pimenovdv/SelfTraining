# Component: Predictive Coding Network (PCN)

**Script:** `train_pcn_component.py`

## Description
This component evaluates a **Predictive Coding Network (PCN)** using pure NumPy. Predictive Coding is a biologically plausible alternative to backpropagation that relies on local learning rules and an iterative inference phase, rather than a global backward pass.

In a PCN, the network maintains both state nodes (values) and error nodes at each layer. The learning process consists of two phases:
1.  **Inference Phase:** The input and target output are clamped. The hidden state nodes are iteratively updated via gradient descent to minimize the local prediction errors (the difference between the state node's value and the top-down prediction from the previous layer).
2.  **Weight Update Phase:** Once the states have settled (or after a fixed number of steps), the weights and biases are updated using a local, Hebbian-like learning rule based solely on the pre-synaptic activations and the post-synaptic errors.

This approach avoids the weight transport problem and non-local credit assignment issues of standard backpropagation, offering insights into how biological brains might perform credit assignment.

## Mathematical Formulation
Let $v_i$ be the state nodes at layer $i$, and $W_i, b_i$ be the weights and biases connecting layer $i$ to $i+1$.
The top-down prediction for layer $i$ is:
$$\mu_i = W_{i-1} \sigma(v_{i-1}) + b_{i-1}$$

The local prediction error at layer $i$ is:
$$e_i = v_i - \mu_i$$

The total network energy is the sum of squared errors:
$$E = \frac{1}{2} \sum_i ||e_i||^2$$

**Inference Phase (Updating $v_i$):**
$$\Delta v_i \propto -\frac{\partial E}{\partial v_i} = -e_i + (W_i^T e_{i+1}) \odot \sigma'(v_i)$$

**Weight Update Phase (Updating $W_i$):**
$$\Delta W_i \propto -\frac{\partial E}{\partial W_i} = e_{i+1} \sigma(v_i)^T$$

## Experiment Results
*   **Task:** Non-linear regression (Sine Wave).
*   **Architecture:** [1, 16, 16, 1]
*   **Result:** The PCN successfully learned to approximate the sine wave, reducing the Mean Squared Error to 0.0009.
*   **Observation:** The local inference-based learning rule was able to effectively train deep representations, demonstrating a viable, biologically motivated alternative to end-to-end backpropagation for representation learning.
