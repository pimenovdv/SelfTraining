import numpy as np
import os

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sample_gumbel(shape, eps=1e-20):
    U = np.random.uniform(0, 1, shape)
    return -np.log(-np.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.shape)
    return softmax(y / temperature)

def train_gumbel_softmax_component(epochs, lr):
    np.random.seed(42)

    # We want to learn logits to output a specific categorical distribution
    # Let's say 4 categories, and we want to target category 2 (one-hot [0, 0, 1, 0])
    target_class = 2
    num_classes = 4

    # Initial logits
    logits = np.random.randn(num_classes)

    target_prob = np.zeros(num_classes)
    target_prob[target_class] = 1.0

    temperature = 1.0
    min_temp = 0.1
    anneal_rate = 0.001

    for epoch in range(epochs):
        # Forward pass
        # Sample using Gumbel-Softmax
        y = gumbel_softmax_sample(logits, temperature)

        # Loss: Cross-entropy with target
        eps = 1e-8
        loss = -np.sum(target_prob * np.log(y + eps))

        # Backward pass
        # dL/dy for cross entropy: -target_prob / y
        dL_dy = -target_prob / (y + eps)

        # Softmax derivative: y * (dL_dy - sum(y * dL_dy))
        dy_dinput = y * (dL_dy - np.sum(y * dL_dy))

        # The input to softmax was (logits + gumbel) / temp
        # So dL/dlogits = dy_dinput / temp
        dL_dlogits = dy_dinput / temperature

        # Update logits
        logits -= lr * dL_dlogits

        # Anneal temperature
        temperature = max(min_temp, temperature * np.exp(-anneal_rate))

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Temp = {temperature:.4f}, Probs = {np.round(softmax(logits), 3)}")

    # Check if we successfully learned to output high probability for target_class
    final_probs = softmax(logits)
    success = final_probs[target_class] > 0.8

    os.makedirs("docs", exist_ok=True)
    report_path = "docs/0079_train_gumbel_softmax_component.md"
    report_content = f"""# Experiment 0079: Train Gumbel-Softmax Component

## Objective
Implement and mathematically model a Gumbel-Softmax estimator, testing the hypothesis that the reparameterization trick with Gumbel noise allows differentiable discrete sampling from a categorical distribution, enabling training via manual backpropagation.

## Setup
*   **Script:** `train_gumbel_softmax_component.py`
*   **Data:** Synthetic target categorical distribution (one-hot).
*   **Hyperparameters:** `epochs` = {epochs}, `learning_rate` = {lr}, `num_classes` = {num_classes}

## Execution
The script was executed to verify the mathematical formulation of Gumbel-Softmax sampling, temperature annealing, and the manual backpropagation of gradients to update underlying logits.

## Results
*   **Status:** {"Success" if success else "Failed"}
*   **Final Loss:** {loss:.4f}
*   **Final Probabilities:** {np.round(final_probs, 4)}
"""
    with open(report_path, "w") as f:
        f.write(report_content)

    return success

if __name__ == "__main__":
    train_gumbel_softmax_component(5000, 0.05)
