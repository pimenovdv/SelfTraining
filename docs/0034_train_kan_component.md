# Experiment 0034: Train Kolmogorov-Arnold Network (KAN) Component

## Objective
To implement and train a Kolmogorov-Arnold Network (KAN) component from scratch using pure `numpy`. KANs represent an alternative to standard MLPs by placing learnable activation functions on the edges (weights) rather than fixed activation functions on the nodes, inspired by the Kolmogorov-Arnold representation theorem.

## Setup
*   **Script:** `train_kan_component.py`
*   **Data:** Synthetic XOR reasoning dataset, a classic test for non-linear capability.
*   **Hyperparameters:** `hidden_dim` = 4, `grid_size` = 5, `epochs` = 10000, `learning_rate` = 0.1
*   **Basis Functions:** Instead of full B-splines, we used a set of Gaussian Radial Basis Functions (RBFs) distributed over a grid on each edge to approximate the 1D univariate functions.

## Execution
The training script was executed successfully.

## Results
*   **Status:** Success.
*   **Convergence:** The model successfully learned the non-linear boundaries of the XOR problem, minimizing the Mean Squared Error over 10000 epochs.
*   **Learning:** Backpropagation accurately computed gradients through the basis functions on the edges using `einsum`, effectively updating the coefficients ($W$) for each basis function across the grid.
*   **Output:** The final predictions closely matched the expected XOR targets.

## Observations & Next Steps
*   This experiment successfully validates the mathematical formulation of placing parameterizable functions on edges.
*   The `numpy` implementation with explicit Einstein summation (`np.einsum`) correctly handles the complex tensor manipulations required for routing gradients to the grid basis coefficients.
*   Future work could explore using true B-splines instead of RBFs for potentially better local control and interpretability, or comparing parameter efficiency between KANs and MLPs of similar representational power.
