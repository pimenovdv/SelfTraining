# Experiment 0017: Train Grouped-Query Attention (GQA) Component

## Objective
To implement and train a small-scale, mathematically rigorous Grouped-Query Attention (GQA) mechanism component of AGI. This serves to test the hypothesis that grouping queries to share key and value heads reduces computational and memory overhead while maintaining high performance, verified via manual forward and backward passes.

## Setup
*   **Script:** `train_gqa_component.py`
*   **Data:** Synthetic sequence dataset.
*   **Hyperparameters:** `d_model` = 4, `num_heads` = 4, `num_kv_heads` = 2, `epochs` = 10, `learning_rate` = 0.1

## Execution
The training script was executed to verify the mathematical formulation of forward and backward passes for the GQA setup.

## Results
*   **Status:** Success.
*   **Loss Reduction:** The model successfully reduced the Mean Squared Error over 10 epochs.
*   **Predictions:** The final predictions closely approximate the expected target outputs.

## Observations & Next Steps
*   The implementation correctly demonstrates the GQA mechanism capabilities and parameter learning. It shows how keys and values can be shared across multiple query heads.
*   Manual derivation of backpropagation using `numpy` solidifies the theoretical understanding of gradient descent for attention grouping. The gradients from grouped queries are successfully aggregated (summed) back into the shared key/value heads.
*   Next steps could involve integrating GQA into a full Transformer Block to benchmark its performance and efficiency compared to standard Multi-Head Attention.
