# Experiment 0117: Train Forward-Forward Component

## Objective
To implement and train a neural network using the Forward-Forward (FF) algorithm, mathematically testing an alternative to backpropagation where layers learn to maximize "goodness" for positive data and minimize it for negative data locally.

## Details
*   **Script:** `train_ff_component.py`
*   **Architecture:** Two local FFLayers with layer normalization and ReLU activation. Goodness is defined as the sum of squared activations.
*   **Training Data:** Synthetic dataset XOR-like problem (positive and negative pairs constructed using one-hot labels).
*   **Learning Rate:** 0.1
*   **Epochs:** 100

## Results
*   **Final Accuracy:** 0.9590
*   **Success:** True

## Conclusion
The Forward-Forward component successfully learned to classify the data using local layer-wise updates without backpropagation, verifying the mathematical feasibility of gradient-free (with respect to subsequent layers) contrastive learning on intermediate representations.
