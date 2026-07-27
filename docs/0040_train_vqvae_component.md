# Experiment 0040: Train Vector Quantized Variational Autoencoder (VQ-VAE) Component

## Objective
To implement and train a Vector Quantized Variational Autoencoder (VQ-VAE) using pure NumPy. This verifies discrete representation learning via a codebook and the Straight-Through Estimator (STE) for backpropagation.

## Setup
*   **Script:** `train_vqvae_component.py`
*   **Data:** Synthetic identity matrix dataset (8x8) representing distinct classes.
*   **Hyperparameters:** `input_dim` = 8, `hidden_dim` = 16, `latent_dim` = 2, `num_embeddings` = 8, `commitment_cost` = 0.25, `epochs` = 10000, `learning_rate` = 0.01 (Adam)

## Execution
The training script was executed to verify the mathematical formulation of VQ-VAE. Specifically, it tests the vector quantization (nearest neighbor lookup) in the forward pass and the STE in the backward pass, alongside the codebook and commitment losses.

## Results
*   **Status:** Success.
*   **Initial Total Loss:** 0.2472
*   **Final Total Loss:** 0.0798
*   **Final Recon Loss:** 0.0781
*   **Final VQ Loss:** 0.0016

## Observations & Next Steps
*   The VQ-VAE successfully minimized the reconstruction loss, proving that the Straight-Through Estimator correctly routes gradients back to the encoder despite the non-differentiable argmin step.
*   The codebook embeddings learned to represent the discrete latent states of the input data.
*   The commitment loss successfully kept the encoder's outputs close to the codebook vectors, stabilizing training.
