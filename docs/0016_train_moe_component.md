# Experiment 0016: Train Mixture of Experts (MoE) Component

## Objective
To implement and train a small-scale, mathematically rigorous Mixture of Experts (MoE) component. This serves to test the hypothesis that a router network can successfully learn to distribute inputs across multiple specialized sub-networks (experts) using basic matrix operations and manual backpropagation.

## Setup
*   **Script:** `train_moe_component.py`
*   **Data:** Synthetic dataset where the target function changes based on the input features, encouraging different experts to specialize.
*   **Hyperparameters:** `num_experts` = 4, `hidden_size` = 8, `epochs` = 10000, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes for both the router and the expert networks. Soft routing (weighted sum of expert outputs by softmax probabilities) was used for differentiable training.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 10000 epochs.
*   **Final Loss:** 0.008375

## Observations & Next Steps
*   The implementation correctly demonstrates the ability of a routing mechanism and multiple experts to jointly learn a complex function.
*   Manual derivation of backpropagation using `numpy.einsum` solidifies the theoretical understanding of gradient flow through the routing probabilities and expert weights.
*   Next steps could involve implementing sparse routing (e.g., Top-1 or Top-2) and investigating load balancing mechanisms (e.g., auxiliary loss for expert utilization) before integrating into a Transformer block.
