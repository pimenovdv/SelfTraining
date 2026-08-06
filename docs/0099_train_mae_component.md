# Experiment: Train Masked Autoencoder (MAE) Component

## Objective
To test the hypothesis that learning robust representations of data can be achieved by masking a significant portion of the input and training a network to reconstruct the missing parts. This forces the model to learn a deep understanding of the underlying structure and dependencies within the data.

## Methodology
A pure mathematical implementation in NumPy of the Masked Autoencoder (MAE) architecture.
1.  **Input:** A sequential dataset where each sample has shape `(L, D)`.
2.  **Masking:** A random `mask_ratio` (e.g., 50%) of the sequence elements are masked out.
3.  **Encoder:** Processes only the unmasked (visible) tokens, along with their positional embeddings. This creates a compact, high-level representation.
4.  **Decoder:** Receives the encoder output tokens and trainable `mask_token`s (with positional embeddings added to all) placed back into their original sequence positions. The decoder attempts to reconstruct the original input values of the masked tokens.
5.  **Loss:** Mean Squared Error (MSE) computed *only* on the masked tokens.
6.  **Optimization:** Adam optimizer updating encoder weights, decoder weights, positional embeddings, and the mask token.

## Results
- **Epochs:** 2000
- **Learning Rate:** 0.005
- **Mask Ratio:** 0.5
- **Final Loss:** 0.000101

## Conclusion
**Success:** The implementation successfully learned to reconstruct the masked portions of the input sequences, significantly reducing the MSE loss over time. This validates the core MAE hypothesis that asymmetric encoder-decoder architectures trained on reconstruction of heavily masked inputs can effectively learn underlying data structures using pure mathematical operations.

**Script:** `train_mae_component.py`
