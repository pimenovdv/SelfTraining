# Component: Predictive Coding Network (PCN) vs Standard Backpropagation Comparison

**Script:** `train_pcn_comparison_component.py`

## Description
This component evaluates and compares a **Predictive Coding Network (PCN)** against a standard **Multi-Layer Perceptron (MLP)** trained with standard Backpropagation on the task of image classification using the digits dataset.

The objective of this experiment is to evaluate the feasibility of PCN on deeper networks and multi-class classification, specifically comparing its accuracy, convergence, and execution time against standard backpropagation.

## Experiment Setup
*   **Task:** Image Classification (Digits dataset, 8x8 pixel images).
*   **Architecture:** 4-layer network `[64, 32, 16, 10]` with Tanh activations.
*   **Training:** Both networks were trained for 20 epochs using simple stochastic updates. The PCN uses an iterative inference phase (10 steps) to settle states before applying local Hebbian updates, while the MLP uses the standard global backward pass.

## Experiment Results
*   **PCN Accuracy:** ~92-94%
*   **MLP Accuracy:** ~97%
*   **PCN Time:** ~13-14 seconds
*   **MLP Time:** ~2.1 seconds

## Conclusion
*   **Accuracy:** The PCN proved capable of learning deep representations for image classification, achieving >90% accuracy, demonstrating that local, biologically plausible learning rules can successfully solve complex, high-dimensional classification problems.
*   **Computational Cost:** The iterative inference phase required by the PCN makes it significantly slower (roughly 6x slower in this pure NumPy implementation) than standard feedforward backpropagation, as the network must settle before weights can be updated.
*   **Observation:** While standard Backpropagation remains more computationally efficient and achieves slightly higher peak accuracy in standard setups, the PCN successfully validates the theoretical capability of local error minimization for deep representation learning.
